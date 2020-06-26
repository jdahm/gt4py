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


class RAJASourceGenerator:
    pass


def raja_layout(mask):
    pass


def raja_is_compatible_layout(field):
    pass


def raja_is_compatible_type(field):
    pass


@gt_backend.register
class DebugBackend(gt_backend.BaseBackend):
    name = "raja-serial"
    options = {}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": raja_layout,
        "is_compatible_layout": raja_is_compatible_layout,
        "is_compatible_type": raja_is_compatible_type,
    }

    GENERATOR_CLASS = RAJAModuleGenerator
