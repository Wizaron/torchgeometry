import pytest

import torch
import torch.nn as nn
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils  # test utils
from utils import check_equal_torch, check_equal_numpy
from common import TEST_DEVICES


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_rad2deg_jit(batch_shape, device_type):
    # define module
    class Rad2Deg(nn.Module):

        def __init__(self):
            super(Rad2Deg, self).__init__()

        def forward(self, x_rad):

            # convert radians/degrees
            x_deg = tgm.rad2deg(x_rad)
            x_deg_to_rad = tgm.deg2rad(x_deg)

            return (x_deg, x_deg_to_rad)

    # initialize module
    rad2deg = Rad2Deg()
    rad2deg = rad2deg.to(torch.device(device_type))

    # trace module
    dummy_input = torch.rand(batch_shape).to(torch.device(device_type))
    rad2deg = torch.jit.trace(rad2deg, dummy_input)

    # generate input data
    x_rad = tgm.pi * torch.rand(batch_shape)
    x_rad = x_rad.to(torch.device(device_type))

    # convert radians/degrees
    x_deg, x_deg_to_rad = rad2deg(x_rad)

    # compute errors
    error = utils.compute_mse(x_rad, x_deg_to_rad)
    assert pytest.approx(error.item(), 0.0)

    # functional
    assert torch.allclose(x_deg, tgm.RadToDeg()(x_rad))


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_deg2rad_jit(batch_shape, device_type):
    # define module
    class Deg2Rad(nn.Module):

        def __init__(self):
            super(Deg2Rad, self).__init__()

        def forward(self, x_deg):

            # convert radians/degrees
            x_rad = tgm.deg2rad(x_deg)
            x_rad_to_deg = tgm.rad2deg(x_rad)

            return (x_rad, x_rad_to_deg)

    # initialize module
    deg2rad = Deg2Rad()
    deg2rad = deg2rad.to(torch.device(device_type))

    # trace module
    dummy_input = torch.rand(batch_shape).to(torch.device(device_type))
    deg2rad = torch.jit.trace(deg2rad, dummy_input)

    # generate input data
    x_deg = 180. * torch.rand(batch_shape)
    x_deg = x_deg.to(torch.device(device_type))

    # convert radians/degrees
    x_rad, x_rad_to_deg = deg2rad(x_deg)

    # compute error
    error = utils.compute_mse(x_deg, x_rad_to_deg)
    assert pytest.approx(error.item(), 0.0)

    # functional
    assert torch.allclose(x_rad, tgm.DegToRad()(x_deg))


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_convert_points_to_homogeneous(batch_shape, device_type):
    # define module
    class Points2Homogeneous(nn.Module):

        def __init__(self):
            super(Points2Homogeneous, self).__init__()

        def forward(self, points):

            # to homogeneous
            points_h = tgm.convert_points_to_homogeneous(points)

            return points_h

    # initialize module
    points2homog = Points2Homogeneous()
    points2homog = points2homog.to(torch.device(device_type))

    # trace module
    dummy_input = torch.rand(batch_shape).to(torch.device(device_type))
    points2homog = torch.jit.trace(points2homog, dummy_input)

    # generate input data
    points = torch.rand(batch_shape)
    points = points.to(torch.device(device_type))

    # to homogeneous
    points_h = points2homog(points)

    assert points_h.shape[-2] == batch_shape[-2]
    assert (points_h[..., -1] == torch.ones(points_h[..., -1].shape)).all()

    # functional
    assert torch.allclose(points_h, tgm.ConvertPointsToHomogeneous()(points))


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_shape", [
    (2, 3), (1, 2, 3), (2, 3, 3), (5, 5, 3), ])
def test_convert_points_from_homogeneous(batch_shape, device_type):
    # define module
    class PointsFromHomogeneous(nn.Module):

        def __init__(self):
            super(PointsFromHomogeneous, self).__init__()

        def forward(self, points_h):

            # to euclidean
            points = tgm.convert_points_from_homogeneous(points_h)

            return points

    # initialize module
    points_from_homog = PointsFromHomogeneous()
    points_from_homog = points_from_homog.to(torch.device(device_type))

    # trace module
    dummy_input = torch.rand(batch_shape).to(torch.device(device_type))
    points_from_homog = torch.jit.trace(points_from_homog, dummy_input)

    # generate input data
    points_h = torch.rand(batch_shape)
    points_h = points_h.to(torch.device(device_type))
    points_h[..., -1] = 1.0

    # to euclidean
    points = points_from_homog(points_h)

    error = utils.compute_mse(points_h[..., :2], points)
    assert pytest.approx(error.item(), 0.0)

    # functional
    assert torch.allclose(points, tgm.ConvertPointsFromHomogeneous()(points_h))


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("num_points", [2, 3, 5])
@pytest.mark.parametrize("num_dims", [2, 3])
def test_transform_points(batch_size, num_points, num_dims, device_type):
    # define module
    class Transform(nn.Module):

        def __init__(self):
            super(Transform, self).__init__()

        def forward(self, points_src, dst_homo_src):

            # transform the points from dst to ref
            points_dst = tgm.transform_points(dst_homo_src, points_src)

            # transform the points from ref to dst
            src_homo_dst = torch.inverse(dst_homo_src)
            points_dst_to_src = tgm.transform_points(src_homo_dst, points_dst)

            return (points_dst, points_dst_to_src)

    eye_size = num_dims + 1

    # initialize module
    point_transformer = Transform()
    point_transformer = point_transformer.to(torch.device(device_type))

    # trace module
    dummy_points_src = torch.rand(batch_size, num_points, num_dims)
    dummy_points_src = dummy_points_src.to(torch.device(device_type))

    dummy_dst_homo_src = utils.create_random_homography(batch_size, eye_size)
    dummy_dst_homo_src = dummy_dst_homo_src.to(torch.device(device_type))

    point_transformer = torch.jit.trace(point_transformer,
                                        (dummy_points_src, dummy_dst_homo_src))

    # generate input data
    points_src = torch.rand(batch_size, num_points, num_dims)
    points_src = points_src.to(torch.device(device_type))

    dst_homo_src = utils.create_random_homography(batch_size, eye_size)
    dst_homo_src = dst_homo_src.to(torch.device(device_type))

    # apply transform
    points_dst, points_dst_to_src = point_transformer(points_src, dst_homo_src)

    # projected should be equal as initial
    error = utils.compute_mse(points_src, points_dst_to_src)
    assert pytest.approx(error.item(), 0.0)

    # functional
    assert torch.allclose(points_dst,
                          tgm.TransformPoints(dst_homo_src)(points_src))


@pytest.mark.parametrize("device_type", TEST_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_angle_axis_to_rotation_matrix(batch_size, device_type):
    # define module
    class AngleAxis2RotMat(nn.Module):

        def __init__(self):
            super(AngleAxis2RotMat, self).__init__()

        def forward(self, angle_axis):

            # apply transform
            rotation_matrix = tgm.angle_axis_to_rotation_matrix(angle_axis)

            rotation_matrix_eye = torch.matmul(
                rotation_matrix, rotation_matrix.transpose(1, 2))

            return (rotation_matrix, rotation_matrix_eye)

    # initialize module
    angle_axis_to_rot_mat = AngleAxis2RotMat()
    angle_axis_to_rot_mat = angle_axis_to_rot_mat.to(torch.device(device_type))

    # trace module
    dummy_input = torch.rand(batch_size, 3)
    dummy_input = dummy_input.to(torch.device(device_type))

    angle_axis_to_rot_mat = torch.jit.trace(angle_axis_to_rot_mat, dummy_input)

    # generate input data
    angle_axis = torch.rand(batch_size, 3).to(torch.device(device_type))

    # apply transform
    rotation_matrix, rotation_matrix_eye = angle_axis_to_rot_mat(angle_axis)

    eye_batch = utils.create_eye_batch(batch_size, 4)
    eye_batch = eye_batch.to(torch.device(device_type))
    assert check_equal_torch(rotation_matrix_eye, eye_batch)


@pytest.mark.parametrize("device_type", TEST_DEVICES)
def test_rotation_matrix_to_angle_axis(device_type):
    # define module
    class RotMat2AngleAxis(nn.Module):

        def __init__(self):
            super(RotMat2AngleAxis, self).__init__()

        def forward(self, rmat):

            return tgm.rotation_matrix_to_angle_axis(rmat)

    device = torch.device(device_type)

    # initialize module
    rot_mat_to_angle_axis = RotMat2AngleAxis()
    rot_mat_to_angle_axis = rot_mat_to_angle_axis.to(device)

    # trace module
    dummy_input = torch.stack([torch.rand(3, 4), torch.rand(3, 4)], dim=0)
    dummy_input = dummy_input.to(device)
    rot_mat_to_angle_axis = torch.jit.trace(rot_mat_to_angle_axis, dummy_input)

    # generate input date
    rmat_1 = torch.tensor([[-0.30382753, -0.95095137, -0.05814062, 0.],
                           [-0.71581715, 0.26812278, -0.64476041, 0.],
                           [0.62872461, -0.15427791, -0.76217038, 0.]])
    rvec_1 = torch.tensor([1.50485376, -2.10737739, 0.7214174])

    rmat_2 = torch.tensor([[0.6027768, -0.79275544, -0.09054801, 0.],
                           [-0.67915707, -0.56931658, 0.46327563, 0.],
                           [-0.41881476, -0.21775548, -0.88157628, 0.]])
    rvec_2 = torch.tensor([-2.44916812, 1.18053411, 0.4085298])
    rmat = torch.stack([rmat_2, rmat_1], dim=0).to(device)
    rvec = torch.stack([rvec_2, rvec_1], dim=0).to(device)

    # apply transform
    angle_axis = rot_mat_to_angle_axis(rmat)

    assert check_equal_torch(angle_axis, rvec)
