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

from gt4py import backend as gt_backend
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils

from .gt_cpu_backend import make_x86_layout_map, x86_is_compatible_layout, gtcpu_is_compatible_type
from .base_gt_backend import BaseGTBackend

class RAJAPyExtGenerator(gt_ir.IRNodeVisitor):



@gt_backend.register
class RajaBackend(gt_backend.BaseBackend):
    GENERATOR_CLASS = RajaPyModuleGenerator
    name = "rajaserial"
    options = BaseGTBackend.GT_BACKEND_OPTS
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": make_x86_layout_map,
        "is_compatible_layout": x86_is_compatible_layout,
        "is_compatible_type": gtcpu_is_compatible_type,
    }

    @classmethod
    def generate_extension(cls, stencil_id, implementation_ir, options):
        pyext_opts = dict(
            verbose=options.backend_opts.pop("verbose", False),
            clean=options.backend_opts.pop("clean", False),
            debug_mode=options.backend_opts.pop("debug_mode", False),
            add_profile_info=options.backend_opts.pop("add_profile_info", False),
        )
