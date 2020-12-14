# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import itertools

import numpy as np
import pytest

import gt4py as gt
from gt4py import backend as gt_backend
from gt4py import gtscript
from gt4py import storage as gt_storage

from ..definitions import ALL_BACKENDS, CPU_BACKENDS, GPU_BACKENDS, INTERNAL_BACKENDS
from .stencil_definitions import EXTERNALS_REGISTRY as externals_registry
from .stencil_definitions import REGISTRY as stencil_definitions


@pytest.mark.parametrize(
    ["name", "backend"], itertools.product(stencil_definitions.names, CPU_BACKENDS)
)
def test_generation_cpu(name, backend):
    stencil_definition = stencil_definitions[name]
    externals = externals_registry[name]
    stencil = gtscript.stencil(backend, stencil_definition, externals=externals, rebuild=True)
    args = {}
    for k, v in stencil_definition.__annotations__.items():
        if isinstance(v, gtscript._FieldDescriptor):
            args[k] = gt_storage.ones(
                dtype=v.dtype,
                mask=gtscript.mask_from_axes(v.axes),
                backend=backend,
                shape=(23, 23, 23),
                default_origin=(10, 10, 10),
            )
        else:
            args[k] = v(1.5)
    stencil(**args, origin=(10, 10, 10), domain=(3, 3, 3))


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    ["name", "backend"], itertools.product(stencil_definitions.names, GPU_BACKENDS)
)
def test_generation_gpu(name, backend):
    stencil_definition = stencil_definitions[name]
    externals = externals_registry[name]
    stencil = gtscript.stencil(backend, stencil_definition, externals=externals)
    args = {}
    for k, v in stencil_definition.__annotations__.items():
        if isinstance(v, gtscript._FieldDescriptor):
            args[k] = gt_storage.ones(
                dtype=v.dtype,
                mask=gtscript.mask_from_axes(v.axes),
                backend=backend,
                shape=(23, 23, 23),
                default_origin=(10, 10, 10),
            )
        else:
            args[k] = v(1.5)
    stencil(**args, origin=(10, 10, 10), domain=(3, 3, 3))


@pytest.mark.requires_gpu
@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_temporary_field_declared_in_if(backend):
    @gtscript.stencil(backend=backend)
    def definition(field_a: gtscript.Field[np.float_]):
        with computation(PARALLEL), interval(...):
            if field_a < 0:
                field_b = -field_a
            else:
                field_b = field_a
            field_a = field_b


@pytest.mark.requires_gpu
@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_stage_without_effect(backend):
    @gtscript.stencil(backend=backend)
    def definition(field_a: gtscript.Field[np.float_]):
        with computation(PARALLEL), interval(...):
            field_c = 0.0


def test_ignore_np_errstate():
    def setup_and_run(backend, **kwargs):
        field_a = gt_storage.zeros(
            dtype=np.float_,
            backend=backend,
            shape=(3, 3, 1),
            default_origin=(0, 0, 0),
        )

        @gtscript.stencil(backend=backend, **kwargs)
        def divide_by_zero(field_a: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                field_a = 1.0 / field_a

        divide_by_zero(field_a)

    # Usual behavior: with the numpy backend there is no error
    setup_and_run(backend="numpy")

    # Expect warning with debug or numpy + ignore_np_errstate=False
    with pytest.warns(RuntimeWarning, match="divide by zero encountered"):
        setup_and_run(backend="debug")

    with pytest.warns(RuntimeWarning, match="divide by zero encountered"):
        setup_and_run(backend="numpy", ignore_np_errstate=False)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_stencil_without_effect(backend):
    def definition1(field_in: gtscript.Field[np.float_]):
        with computation(PARALLEL), interval(...):
            tmp = 0.0

    def definition2(f_in: gtscript.Field[np.float_]):
        from __externals__ import flag
        from __gtscript__ import __INLINE

        with computation(PARALLEL), interval(...):
            if __INLINED(flag):
                B = f_in

    stencil1 = gtscript.stencil(backend, definition1)
    stencil2 = gtscript.stencil(backend, definition2, externals={"flag": False})

    field_in = gt_storage.ones(
        dtype=np.float_, backend=backend, shape=(23, 23, 23), default_origin=(0, 0, 0)
    )

    # test with explicit domain specified
    stencil1(field_in, domain=(3, 3, 3))
    stencil2(field_in, domain=(3, 3, 3))

    # test without domain specified
    with pytest.raises(ValueError):
        stencil1(field_in)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_stage_merger_induced_interval_block_reordering(backend):
    field_in = gt_storage.ones(
        dtype=np.float_, backend=backend, shape=(23, 23, 23), default_origin=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float_, backend=backend, shape=(23, 23, 23), default_origin=(0, 0, 0)
    )

    @gtscript.stencil(backend=backend)
    def stencil(field_in: gtscript.Field[np.float_], field_out: gtscript.Field[np.float_]):
        with computation(BACKWARD):
            with interval(-2, -1):  # block 1
                field_out = field_in
            with interval(0, -2):  # block 2
                field_out = field_in
        with computation(BACKWARD):
            with interval(-1, None):  # block 3
                field_out = 2 * field_in
            with interval(0, -1):  # block 4
                field_out = 3 * field_in

    stencil(field_in, field_out)

    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, 0:-1], 3)
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, -1], 2)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_1d_fields(backend):
    Field3D = gtscript.Field[np.float_]
    Field1D = gtscript.Field[np.float_, gtscript.K]

    @gtscript.stencil(backend=backend)
    def k_field_stencil(in_field: Field3D, out_field: Field3D, sum1: Field1D):
        with computation(FORWARD), interval(...):
            if in_field + sum1 > 0.0:
                sum1 = sum1[-1] + in_field
        with computation(BACKWARD), interval(...):
            if in_field < 0.0 and sum1 >= 0.0:
                out_field = sum1 if sum1 < in_field else in_field

    in_field = gt_storage.ones(
        dtype=np.float_, backend=backend, shape=(23, 23, 23), default_origin=(0, 0, 0)
    )
    out_field = gt_storage.ones(
        dtype=np.float_, backend=backend, shape=(23, 23, 23), default_origin=(0, 0, 0)
    )
    sum_field = gt_storage.zeros(
        dtype=np.float_,
        backend=backend,
        shape=(23,),
        default_origin=(1,),
        mask=(False, False, True),
    )

    k_field_stencil(in_field, out_field, sum_field, origin=(0, 0, 0))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_2d_fields(backend):
    Field3D = gtscript.Field[np.float_]
    Field2D = gtscript.Field[np.float_, gtscript.IJ]

    @gtscript.stencil(backend=backend)
    def vertical_sum(nums: Field3D, sums: Field2D):
        with computation(FORWARD), interval(...):
            sums += nums

    nums = gt_storage.ones(
        dtype=np.float_, backend=backend, shape=(23, 23, 23), default_origin=(0, 0, 0)
    )
    sums = gt_storage.zeros(
        dtype=np.float_,
        backend=backend,
        shape=(23, 23),
        default_origin=(0, 0),
        mask=(True, True, False),
    )

    vertical_sum(nums, sums, origin=(0, 0, 0))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_2d_parallel_if(backend):
    FloatD = gtscript.Field[np.float_]
    Int2D = gtscript.Field[np.int, gtscript.IJ]

    @gtscript.stencil(backend=backend)
    def neg_z_counter(q: FloatD, zcnt: Int2D):
        with computation(PARALLEL), interval(...):
            if q < 0.0:
                zcnt += 1

    q = gt_storage.ones(
        dtype=np.float_, backend=backend, shape=(5, 5, 5), default_origin=(0, 0, 0)
    )
    q[1::4, ::4] = -1
    q[::4, 1::4] = -1
    zcnt = gt_storage.zeros(
        dtype=np.int,
        backend=backend,
        shape=(5, 5),
        default_origin=(0, 0),
        mask=(True, True, False),
    )

    # neg_z_counter(q, zcnt, domain=(5, 5, 5), origin=(0, 0, 0))
    neg_z_counter(q, zcnt, origin=(0, 0, 0))


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_lower_dimensional_inputs(backend):
    @gtscript.stencil(backend=backend)
    def stencil(
        field_3d: gtscript.Field[np.float_, gtscript.IJK],
        field_2d: gtscript.Field[np.float_, gtscript.IJ],
        field_1d: gtscript.Field[np.float_, gtscript.K],
    ):
        with computation(PARALLEL), interval(0, 1):
            field_2d = field_1d + field_3d

        with computation(PARALLEL):
            with interval(0, 1):
                tmp = field_2d + field_1d
                field_2d = tmp[1, 0, 0] + field_1d
            with interval(1, None):
                field_3d = tmp[-1, 0, 0]

    full_shape = (6, 6, 3)
    default_origin = (1, 1, 0)
    dtype = float

    field_3d = gt_storage.ones(backend, default_origin, full_shape, dtype, mask=None)
    field_2d = gt_storage.ones(
        backend, default_origin[:-1], full_shape[:-1], dtype, mask=(True, True, False)
    )
    field_1d = gt_storage.ones(
        backend, (default_origin[-1],), (full_shape[-1],), dtype, mask=(False, False, True)
    )

    stencil(field_3d, field_2d, field_1d, origin=(1, 1, 0))
