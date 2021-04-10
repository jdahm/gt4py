# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

import abc
import copy
import functools
import numbers
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Optional,
    OrderedDict,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import jinja2
import numpy as np

from gt4py import analysis as gt_analysis
from gt4py import backend as gt_backend
from gt4py import definitions as gt_definitions
from gt4py import gt_src_manager
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils
from gt4py.utils import text as gt_text
from gt4py.utils.attrib import Any as AnyT
from gt4py.utils.attrib import Dict as DictOf
from gt4py.utils.attrib import List as ListOf
from gt4py.utils.attrib import Set as SetOf
from gt4py.utils.attrib import attribkwclass as attribclass
from gt4py.utils.attrib import attribute

from . import pyext_builder


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject
    from gt4py.storage.storage import Storage


def make_x86_layout_map(mask: Tuple[int, ...]) -> Tuple[Optional[int], ...]:
    ctr = iter(range(sum(mask)))
    if len(mask) < 3:
        layout: List[Optional[int]] = [next(ctr) if m else None for m in mask]
    else:
        swapped_mask: List[Optional[int]] = [*mask[3:], *mask[:3]]
        layout = [next(ctr) if m else None for m in swapped_mask]

        layout = [*layout[-3:], *layout[:-3]]

    return tuple(layout)


def x86_is_compatible_layout(field: "Storage") -> bool:
    stride = 0
    layout_map = make_x86_layout_map(field.mask)
    flattened_layout = [index for index in layout_map if index is not None]
    if len(field.strides) < len(flattened_layout):
        return False
    for dim in reversed(np.argsort(flattened_layout)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


def gtcpu_is_compatible_type(field: "Storage") -> bool:
    return isinstance(field, np.ndarray)


def make_mc_layout_map(mask: Tuple[int, ...]) -> Tuple[Optional[int], ...]:
    ctr = reversed(range(sum(mask)))
    if len(mask) < 3:
        layout: List[Optional[int]] = [next(ctr) if m else None for m in mask]
    else:
        swapped_mask: List[Optional[int]] = list(mask)
        tmp = swapped_mask[1]
        swapped_mask[1] = swapped_mask[2]
        swapped_mask[2] = tmp

        layout = [next(ctr) if m else None for m in swapped_mask]

        tmp = layout[1]
        layout[1] = layout[2]
        layout[2] = tmp

    return tuple(layout)


def mc_is_compatible_layout(field: "Storage") -> bool:
    stride = 0
    layout_map = make_mc_layout_map(field.mask)
    flattened_layout = [index for index in layout_map if index is not None]
    if len(field.strides) < len(flattened_layout):
        return False
    for dim in reversed(np.argsort(flattened_layout)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


def cuda_layout(mask: Tuple[int, ...]) -> Tuple[Optional[int], ...]:
    ctr = reversed(range(sum(mask)))
    return tuple([next(ctr) if m else None for m in mask])


def cuda_is_compatible_layout(field: "Storage") -> bool:
    stride = 0
    layout_map = cuda_layout(field.mask)
    flattened_layout = [index for index in layout_map if index is not None]
    if len(field.strides) < len(flattened_layout):
        return False
    for dim in reversed(np.argsort(flattened_layout)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


def cuda_is_compatible_type(field: Any) -> bool:
    from gt4py.storage.storage import ExplicitlySyncedGPUStorage, GPUStorage

    return isinstance(field, (GPUStorage, ExplicitlySyncedGPUStorage))


class GTNode(gt_ir.IIRNode):
    pass


@attribclass
class Computation(GTNode):
    multistages = attribute(of=ListOf[gt_ir.MultiStage])
    api_fields = attribute(of=SetOf[str])
    parameters = attribute(of=SetOf[str])
    arg_fields = attribute(of=SetOf[str])
    tmp_fields = attribute(of=DictOf[str, gt_ir.FieldDecl])


@attribclass
class GTStencil(GTNode):
    name = attribute(of=str)
    domain = attribute(of=gt_ir.Domain)
    computations = attribute(of=ListOf[Computation])
    api_signature = attribute(of=ListOf[gt_ir.ArgumentInfo])
    fields = attribute(of=DictOf[str, gt_ir.FieldDecl])  # api and arg decls
    parameters = attribute(of=DictOf[str, gt_ir.VarDecl])
    fields_extents = attribute(of=DictOf[str, gt_ir.Extent])
    unreferenced = attribute(of=ListOf[str], factory=list)
    externals = attribute(of=DictOf[str, AnyT], optional=True)

    @property
    def has_effect(self):
        """
        Determine whether the stencil modifies any of its arguments.

        Note that the only guarantee of this function is that the stencil has no effect if it returns ``false``. It
        might however return true in cases where the optimization passes were not able to deduce this.
        """
        for computation in self.computations:
            if any(field not in self.unreferenced for field in computation.api_fields):
                return True


class LowerHorizontalIf(gt_ir.IRNodeMapper):
    """Replaces gt_ir.HorizontalIf with gt_ir.If, gt_ir.AxisOffset, gt_ir.AxisIndex."""

    @classmethod
    def apply(cls, impl_node: GTStencil) -> None:
        cls(impl_node.domain).visit(impl_node)

    def __init__(self, domain: gt_ir.Domain):
        self.domain = domain
        self.extent: Optional[gt_ir.Extent] = None

    def visit_Stage(
        self, path: tuple, node_name: str, node: gt_ir.Stage
    ) -> Tuple[bool, gt_ir.Stage]:
        self.extent = node.compute_extent
        return self.generic_visit(path, node_name, node)

    def visit_HorizontalIf(
        self, path: tuple, node_name: str, node: gt_ir.HorizontalIf
    ) -> Tuple[bool, gt_ir.If]:
        assert self.extent is not None

        conditions = []
        for axis, interval in node.intervals.items():
            extent = self.extent[self.domain.index(axis)]

            if (
                interval.start.level == interval.end.level
                and interval.start.offset == interval.end.offset - 1
            ):
                # Use a single condition
                conditions.append(
                    gt_ir.BinOpExpr(
                        op=gt_ir.BinaryOperator.EQ,
                        lhs=gt_ir.AxisIndex(axis=axis),
                        rhs=gt_ir.AxisOffset(
                            axis=axis, endpt=interval.start.level, offset=interval.start.offset
                        ),
                    )
                )
            else:
                # start
                if (
                    interval.start.level != gt_ir.LevelMarker.START
                    or interval.start.offset > extent[0]
                ):
                    conditions.append(
                        gt_ir.BinOpExpr(
                            op=gt_ir.BinaryOperator.GE,
                            lhs=gt_ir.AxisIndex(axis=axis),
                            rhs=gt_ir.AxisOffset(
                                axis=axis, endpt=interval.start.level, offset=interval.start.offset
                            ),
                        )
                    )

                # end
                if interval.end.level != gt_ir.LevelMarker.END or interval.end.offset < extent[1]:
                    conditions.append(
                        gt_ir.BinOpExpr(
                            op=gt_ir.BinaryOperator.LT,
                            lhs=gt_ir.AxisIndex(axis=axis),
                            rhs=gt_ir.AxisOffset(
                                axis=axis, endpt=interval.end.level, offset=interval.end.offset
                            ),
                        )
                    )

        if conditions:
            return (
                True,
                gt_ir.If(
                    condition=functools.reduce(
                        lambda x, y: gt_ir.BinOpExpr(op=gt_ir.BinaryOperator.AND, lhs=x, rhs=y),
                        conditions,
                    ),
                    main_body=node.body,
                ),
            )
        else:
            return True, node.body


class LowerToGTStencil(gt_ir.IRNodeMapper):
    """Lower StencilImplementation to a GTStencil.

    1. Place each MultiStage into its own Computation
    2. Fill arg_fields with fields used inside HorizontalIf nodes

    The IR will be invalid after this mostly naive lowering. To become valid,
    it will need the remainder of the passes in :func:`impl_to_gtstencil`.
    """

    @classmethod
    def apply(cls, impl_node: gt_ir.StencilImplementation, allocated_fields: Set[str]) -> None:
        return cls(impl_node, allocated_fields).visit(impl_node)

    def __init__(self, impl_node: gt_ir.StencilImplementation, allocated_fields: Set[str]):
        self.impl_node = impl_node
        self.allocated_fields = allocated_fields

    def __call__(self, impl_node: gt_ir.StencilImplementation):
        return self.visit(impl_node)

    @staticmethod
    def stages_in_multistage(multi_stage: gt_ir.MultiStage) -> Generator[gt_ir.Stage, None, None]:
        for group in multi_stage.groups:
            yield from group.stages

    def visit_MultiStage(self, path: tuple, node_name: str, node: gt_ir.MultiStage) -> Computation:
        api_signature_names = [arg.name for arg in self.impl_node.api_signature]

        api_fields = set()
        arg_fields = set()
        parameters = set()
        tmp_fields = dict()
        for stage in self.stages_in_multistage(node):
            api_fields.update(
                {
                    accessor.symbol
                    for accessor in stage.accessors
                    if isinstance(accessor, gt_ir.FieldAccessor)
                    and accessor.symbol in api_signature_names
                }
            )
            arg_fields.update(
                {
                    accessor.symbol
                    for accessor in stage.accessors
                    if isinstance(accessor, gt_ir.FieldAccessor)
                    and accessor.symbol in self.allocated_fields
                    and accessor.symbol not in api_signature_names
                }
            )
            parameters.update(
                {
                    accessor.symbol
                    for accessor in stage.accessors
                    if isinstance(accessor, gt_ir.ParameterAccessor)
                }
            )
            tmp_names = {
                accessor.symbol
                for accessor in stage.accessors
                if isinstance(accessor, gt_ir.FieldAccessor)
                and accessor.symbol not in api_signature_names
                and accessor.symbol not in self.allocated_fields
            }
            tmp_fields.update({name: self.impl_node.fields[name] for name in tmp_names})

        return True, Computation(
            multistages=[node],
            api_fields=api_fields,
            arg_fields=arg_fields,
            parameters=parameters,
            tmp_fields=tmp_fields,
        )

    def visit_StencilImplementation(
        self, path: tuple, node_name: str, node: gt_ir.StencilImplementation
    ) -> GTStencil:
        computations = [self.visit(multi_stage) for multi_stage in node.multi_stages]

        tmp_fields = set().union(*(computation.tmp_fields for computation in computations))
        api_and_arg_fields = {
            name: decl for name, decl in node.fields.items() if name not in tmp_fields
        }

        return True, GTStencil(
            name=node.name,
            domain=node.domain,
            computations=computations,
            api_signature=node.api_signature,
            fields=api_and_arg_fields,
            parameters=node.parameters,
            fields_extents=node.fields_extents,
            unreferenced=node.unreferenced,
            externals=node.externals,
        )


class ComputationMergingWrapper:
    def __init__(self, computation: Computation):
        self._computation = computation

    @classmethod
    def wrap_items(
        cls,
        items: Sequence[Computation],
    ) -> List["ComputationMergingWrapper"]:
        return [cls(block) for block in items]

    def can_merge_with(self, candidate: "ComputationMergingWrapper") -> bool:
        candidate_allocated_inputs = {
            field
            for field in candidate.field_accessors_with_intent(
                gt_ir.AccessIntent.READ, has_nonzero_parallel_extent=True
            )
            if field in candidate.arg_fields | candidate.api_fields
        }

        self_allocated_outputs = {
            field
            for field in self.field_accessors_with_intent(gt_ir.AccessIntent.WRITE)
            if field in self.arg_fields | self.api_fields
        }
        return not candidate_allocated_inputs.intersection(self_allocated_outputs)

    def merge_with(self, candidate: "ComputationMergingWrapper") -> None:
        self.computation.multistages.extend(candidate.computation.multistages)
        self.computation.api_fields.update(candidate.computation.api_fields)
        self.computation.arg_fields.update(candidate.computation.arg_fields)
        self.computation.parameters.update(candidate.computation.parameters)
        self.computation.tmp_fields.update(candidate.computation.tmp_fields)

    @property
    def stages(self) -> Generator[gt_ir.Stage, None, None]:
        for multistage in self.computation.multistages:
            for group in multistage.groups:
                yield from group.stages

    def field_accessors_with_intent(self, intent, has_nonzero_parallel_extent=False):
        fields = set()
        for stage in self.stages:
            fields |= {
                accessor.symbol
                for accessor in stage.accessors
                if isinstance(accessor, gt_ir.FieldAccessor)
                and bool(accessor.intent & intent)
                and (
                    not has_nonzero_parallel_extent
                    or not gt_definitions.Extent(accessor.extent[:-1]).is_zero
                )
            }
        return fields

    @property
    def computation(self) -> Computation:
        return self._computation

    @property
    def arg_fields(self) -> Set[str]:
        return self.computation.arg_fields

    @property
    def api_fields(self) -> Set[str]:
        return self.computation.api_fields

    @property
    def wrapped(self) -> Computation:
        return self._computation


class UpdateArgFields:
    @property
    def stages_with_computation_index(self) -> Generator[gt_ir.Stage, None, None]:
        for comp_index, computation in enumerate(self.computations):
            yield from (
                (comp_index, stage)
                for multistage in computation.multistages
                for group in multistage.groups
                for stage in group.stages
            )

    @property
    def computations(self) -> Generator[Computation, None, None]:
        return self.gtstencil.computations

    @property
    def gtstencil(self) -> GTStencil:
        return self._gtstencil

    def __init__(self, gtstencil: GTStencil):
        self._gtstencil = gtstencil

    def __call__(self):
        # Collect set of all arg_fields to propagate to all computations
        promote_to_arg_fields = set().union(
            *(comp.arg_fields for comp in self.gtstencil.computations)
        )
        field_to_last_write_loc = {}

        # Promote temporaries that are READ before any WRITE in the same computation.
        # Also, READ in another computation than WRITE.
        for comp_index, stage in self.stages_with_computation_index:
            computation_tmp_fields = self.gtstencil.computations[comp_index].tmp_fields
            for symbol in {
                accessor.symbol
                for accessor in stage.accessors
                if isinstance(accessor, gt_ir.FieldAccessor)
                and bool(accessor.intent & gt_ir.AccessIntent.READ)
                and accessor.symbol in computation_tmp_fields
            }:
                last_write_comp = field_to_last_write_loc.get(symbol, None)
                if last_write_comp is not None and last_write_comp != comp_index:
                    promote_to_arg_fields.add(symbol)

            for symbol in {
                accessor.symbol
                for accessor in stage.accessors
                if isinstance(accessor, gt_ir.FieldAccessor)
                and bool(accessor.intent & gt_ir.AccessIntent.WRITE)
            }:
                field_to_last_write_loc[symbol] = comp_index

        # In each computation, move requested fields from temporaries to arg_fields
        for computation in self.gtstencil.computations:
            to_move = {
                tmp: value
                for tmp, value in computation.tmp_fields.items()
                if tmp in promote_to_arg_fields
            }
            computation.arg_fields.update(to_move.keys())
            # Add FieldDecl value to gtstencil.fields
            self.gtstencil.fields.update(to_move)
            for tmp in to_move:
                del computation.tmp_fields[tmp]

    @classmethod
    def apply(cls, gtstencil: GTStencil):
        cls(gtstencil)()


def impl_to_gtstencil(impl_node: gt_ir.StencilImplementation) -> GTStencil:
    allocated_fields = {
        name
        for name, decl in impl_node.fields.items()
        if isinstance(decl, gt_ir.FieldDecl) and decl.requires_sync
    }

    # Trivially lower StencilImplementation to GTStencil
    gtstencil = LowerToGTStencil.apply(copy.deepcopy(impl_node), allocated_fields)

    # Merge computations as possible
    gtstencil.computations = gt_analysis.passes.greedy_merging_with_wrapper(
        gtstencil.computations, ComputationMergingWrapper
    )

    # Convert HorizontalIf -> If
    LowerHorizontalIf.apply(gtstencil)

    # Propagate arg_fields across computations and promote temporaries as needed
    UpdateArgFields.apply(gtstencil)

    return gtstencil


class _MaxKOffsetExtractor(gt_ir.IRNodeVisitor):
    @classmethod
    def apply(cls, root_node: gt_ir.Node) -> int:
        return cls()(root_node)

    def __init__(self):
        self.max_offset = 2

    def __call__(self, node: gt_ir.Node) -> int:
        self.visit(node)
        return self.max_offset

    def visit_AxisBound(self, node: gt_ir.AxisBound) -> None:
        self.max_offset = max(self.max_offset, abs(node.offset) + 1)


_extract_max_k_offset = _MaxKOffsetExtractor.apply


class GTPyExtGenerator(gt_ir.IRNodeVisitor):

    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
    TEMPLATE_FILES = {
        "computation.hpp": "computation.hpp.in",
        "computation.src": "computation.src.in",
        "bindings.cpp": "bindings.cpp.in",
    }
    COMPUTATION_FILES = ["computation.hpp", "computation.src"]
    BINDINGS_FILES = ["bindings.cpp"]

    OP_TO_CPP = {
        gt_ir.UnaryOperator.POS: "+",
        gt_ir.UnaryOperator.NEG: "-",
        gt_ir.UnaryOperator.NOT: "!",
        gt_ir.BinaryOperator.ADD: "+",
        gt_ir.BinaryOperator.SUB: "-",
        gt_ir.BinaryOperator.MUL: "*",
        gt_ir.BinaryOperator.DIV: "/",
        gt_ir.BinaryOperator.POW: lambda lhs, rhs: "pow({lhs}, {rhs})".format(lhs=lhs, rhs=rhs),
        gt_ir.BinaryOperator.AND: "&&",
        gt_ir.BinaryOperator.OR: "||",
        gt_ir.BinaryOperator.LT: "<",
        gt_ir.BinaryOperator.LE: "<=",
        gt_ir.BinaryOperator.EQ: "==",
        gt_ir.BinaryOperator.GE: ">=",
        gt_ir.BinaryOperator.GT: ">",
        gt_ir.BinaryOperator.NE: "!=",
    }

    DATA_TYPE_TO_CPP = {
        gt_ir.DataType.BOOL: "bool",
        gt_ir.DataType.INT8: "int8_t",
        gt_ir.DataType.INT16: "int16_t",
        gt_ir.DataType.INT32: "int32_t",
        gt_ir.DataType.INT64: "int64_t",
        gt_ir.DataType.FLOAT32: "float32_t",
        gt_ir.DataType.FLOAT64: "float64_t",
        gt_ir.DataType.DEFAULT: "float64_t",
    }

    NATIVE_FUNC_TO_CPP = {
        gt_ir.NativeFunction.ABS: "fabs",
        gt_ir.NativeFunction.MIN: "min",
        gt_ir.NativeFunction.MAX: "max",
        gt_ir.NativeFunction.MOD: "fmod",
        gt_ir.NativeFunction.SIN: "sin",
        gt_ir.NativeFunction.COS: "cos",
        gt_ir.NativeFunction.TAN: "tan",
        gt_ir.NativeFunction.ARCSIN: "asin",
        gt_ir.NativeFunction.ARCCOS: "acos",
        gt_ir.NativeFunction.ARCTAN: "atan",
        gt_ir.NativeFunction.SQRT: "sqrt",
        gt_ir.NativeFunction.EXP: "exp",
        gt_ir.NativeFunction.LOG: "log",
        gt_ir.NativeFunction.ISFINITE: "isfinite",
        gt_ir.NativeFunction.ISINF: "isinf",
        gt_ir.NativeFunction.ISNAN: "isnan",
        gt_ir.NativeFunction.FLOOR: "floor",
        gt_ir.NativeFunction.CEIL: "ceil",
        gt_ir.NativeFunction.TRUNC: "trunc",
    }

    BUILTIN_TO_CPP = {
        gt_ir.Builtin.NONE: "nullptr",  # really?
        gt_ir.Builtin.FALSE: "false",
        gt_ir.Builtin.TRUE: "true",
    }

    ITERATION_ORDER_TO_GT_ORDER = {
        gt_ir.IterationOrder.FORWARD: "forward",
        gt_ir.IterationOrder.BACKWARD: "backward",
        gt_ir.IterationOrder.PARALLEL: "forward",  # NOTE requires sync between parallel mss
    }

    def __init__(self, class_name, module_name, gt_backend_t, options):
        self.class_name = class_name
        self.module_name = module_name
        self.gt_backend_t = gt_backend_t
        self.options = options

        self.templates = {}
        for key, file_name in self.TEMPLATE_FILES.items():
            with open(os.path.join(self.TEMPLATE_DIR, file_name), "r") as f:
                self.templates[key] = jinja2.Template(f.read())
        self.impl_node = None
        self.stage_symbols = None
        self.apply_block_symbols = None
        self.declared_symbols = None

    def __call__(self, impl_node: gt_ir.StencilImplementation) -> Dict[str, Dict[str, str]]:
        assert isinstance(impl_node, gt_ir.StencilImplementation)
        assert impl_node.domain.sequential_axis.name == gt_definitions.CartesianSpace.Axis.K.name

        self.impl_node = impl_node

        self.domain = impl_node.domain
        self.k_splitters: List[Tuple[str, int]] = []

        gtstencil = impl_to_gtstencil(self.impl_node)
        source = self.visit(gtstencil)

        return source

    def _make_cpp_value(self, value: Any) -> Optional[str]:
        if isinstance(value, numbers.Number):
            if isinstance(value, bool):
                value = int(value)
                result: Optional[str] = str(value)
        else:
            result = None

        return result

    def _make_cpp_type(self, data_type: gt_ir.DataType) -> str:
        result = self.DATA_TYPE_TO_CPP[data_type]

        return result

    def _make_cpp_variable(self, decl: gt_ir.VarDecl) -> str:
        result = "{t} {name};".format(t=self._make_cpp_type(decl.data_type), name=decl.name)

        return result

    def visit_ScalarLiteral(self, node: gt_ir.ScalarLiteral) -> str:
        source = "{dtype}{{{value}}}".format(
            dtype=self.DATA_TYPE_TO_CPP[node.data_type], value=node.value
        )

        return source

    def visit_FieldRef(self, node: gt_ir.FieldRef, **kwargs: Any) -> str:
        assert node.name in self.apply_block_symbols
        offset = [node.offset.get(name, 0) for name in self.domain.axes_names]
        if not all(i == 0 for i in offset):
            idx = ", ".join(str(i) for i in offset)
        else:
            idx = ""
        source = "eval({name}({idx}))".format(name=node.name, idx=idx)

        return source

    def visit_VarRef(self, node: gt_ir.VarRef, *, write_context: bool = False) -> str:
        assert node.name in self.apply_block_symbols

        if write_context and node.name not in self.declared_symbols:
            self.declared_symbols.add(node.name)
            source = self._make_cpp_type(self.apply_block_symbols[node.name].data_type) + " "
        else:
            source = ""

        idx = ", ".join(str(i) for i in node.index) if node.index else ""

        if node.name in self.impl_node.parameters:
            source += "eval({name}({idx}))".format(name=node.name, idx=idx)
        else:
            source += "{name}".format(name=node.name)
            if idx:
                source += "[{idx}]".format(idx=idx)

        return source

    def visit_UnaryOpExpr(self, node: gt_ir.UnaryOpExpr) -> str:
        fmt = "({})" if isinstance(node.arg, gt_ir.CompositeExpr) else "{}"
        source = "{op}{expr}".format(
            op=self.OP_TO_CPP[node.op], expr=fmt.format(self.visit(node.arg))
        )

        return source

    def visit_BinOpExpr(self, node: gt_ir.BinOpExpr) -> str:
        lhs_fmt = "({})" if isinstance(node.lhs, gt_ir.CompositeExpr) else "{}"
        lhs_expr = lhs_fmt.format(self.visit(node.lhs))
        rhs_fmt = "({})" if isinstance(node.rhs, gt_ir.CompositeExpr) else "{}"
        rhs_expr = rhs_fmt.format(self.visit(node.rhs))

        cpp_op = self.OP_TO_CPP[node.op]
        if callable(cpp_op):
            source = cpp_op(lhs_expr, rhs_expr)
        else:
            source = "{lhs} {op} {rhs}".format(lhs=lhs_expr, op=cpp_op, rhs=rhs_expr)

        return source

    def visit_Cast(self, node: gt_ir.Cast) -> str:
        expr = self.visit(node.expr)
        dtype = self.DATA_TYPE_TO_CPP[node.data_type]
        return f"static_cast<{dtype}>({expr})"

    def visit_BuiltinLiteral(self, node: gt_ir.BuiltinLiteral) -> str:
        return self.BUILTIN_TO_CPP[node.value]

    def visit_NativeFuncCall(self, node: gt_ir.NativeFuncCall) -> str:
        call = self.NATIVE_FUNC_TO_CPP[node.func]
        if self.gt_backend_t != "cuda":
            call = "std::" + call
        args = ",".join(self.visit(arg) for arg in node.args)
        return f"{call}({args})"

    def visit_TernaryOpExpr(self, node: gt_ir.TernaryOpExpr) -> str:
        then_fmt = "({})" if isinstance(node.then_expr, gt_ir.CompositeExpr) else "{}"
        else_fmt = "({})" if isinstance(node.else_expr, gt_ir.CompositeExpr) else "{}"
        source = "({condition}) ? {then_expr} : {else_expr}".format(
            condition=self.visit(node.condition),
            then_expr=then_fmt.format(self.visit(node.then_expr)),
            else_expr=else_fmt.format(self.visit(node.else_expr)),
        )

        return source

    def visit_Assign(self, node: gt_ir.Assign) -> List[str]:
        lhs = self.visit(node.target, write_context=True)
        rhs = self.visit(node.value)
        source = "{lhs} = {rhs};".format(lhs=lhs, rhs=rhs)

        return [source]

    def visit_BlockStmt(self, node: gt_ir.BlockStmt) -> str:
        body_sources = gt_text.TextBlock()
        for stmt in node.stmts:
            body_sources.extend(self.visit(stmt))

        return body_sources.text

    def visit_If(self, node: gt_ir.If) -> gt_text.TextBlock:
        body_sources = gt_text.TextBlock()
        body_sources.append("if ({condition}) {{".format(condition=self.visit(node.condition)))
        for stmt in node.main_body.stmts:
            body_sources.extend(self.visit(stmt))
        if node.else_body:
            body_sources.append("} else {")

            for stmt in node.else_body.stmts:
                body_sources.extend(self.visit(stmt))

        body_sources.append("}")
        return body_sources

    def visit_AxisBound(self, node: gt_ir.AxisBound) -> Tuple[int, int]:
        if node.level == gt_ir.LevelMarker.START:
            level = 0
        elif node.level == gt_ir.LevelMarker.END:
            level = len(self.k_splitters) + 1
        else:
            raise NotImplementedError("VarRefs are not yet supported")

        # Shift offset to make it relative to the splitter (in-between levels)
        offset = node.offset + 1 if node.offset >= 0 else node.offset

        return level, offset

    def visit_AxisInterval(
        self, node: gt_ir.AxisInterval
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        start_splitter, start_offset = self.visit(node.start)
        end_splitter, end_offset = self.visit(node.end)

        # Transform range from excluded endpoint to including endpoint
        end_offset = -1 if end_offset == 1 else end_offset - 1

        return (start_splitter, start_offset), (end_splitter, end_offset)

    def visit_AxisIndex(self, node: gt_ir.AxisIndex) -> str:
        return f"eval.{node.axis.lower()}()"

    def visit_AxisOffset(self, node: gt_ir.AxisOffset) -> str:
        return "static_cast<gt::int_t>({endpt}{offset:+d})".format(
            endpt=f"eval(domain_size_{node.axis.upper()}())"
            if node.endpt == gt_ir.LevelMarker.END
            else "0",
            offset=node.offset,
        )

    def visit_ApplyBlock(
        self, node: gt_ir.ApplyBlock
    ) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], str]:
        interval_definition = self.visit(node.interval)

        body_sources = gt_text.TextBlock()

        self.declared_symbols = set()
        for name, var_decl in node.local_symbols.items():
            assert isinstance(var_decl, gt_ir.VarDecl)
            body_sources.append(self._make_cpp_variable(var_decl))
            self.declared_symbols.add(name)

        self.apply_block_symbols = {**self.stage_symbols, **node.local_symbols}
        body_sources.extend(self.visit(node.body))

        return interval_definition, body_sources.text

    def visit_Stage(self, node: gt_ir.Stage):
        # Initialize symbols for the generation of references in this stage
        self.stage_symbols = {}
        args = []
        for accessor in node.accessors:
            self.stage_symbols[accessor.symbol] = accessor
            arg = {"name": accessor.symbol, "access_type": "in", "extent": None}
            if isinstance(accessor, gt_ir.FieldAccessor):
                # Both WRITE and READ_WRITE map to "inout"
                arg["access_type"] = "in" if accessor.intent == gt_ir.AccessIntent.READ else "inout"
                arg["extent"] = gt_utils.flatten(accessor.extent)
            args.append(arg)

        for name in sorted(
            list({node.axis for node in gt_ir.filter_nodes_dfs(node, gt_ir.AxisIndex)})
        ):
            args.append({"name": f"domain_size_{name}", "access_type": "in", "extent": None})

        # Create regions and computations
        regions = []
        for apply_block in node.apply_blocks:
            interval_definition, body_sources = self.visit(apply_block)
            regions.append(
                {
                    "interval_start": interval_definition[0],
                    "interval_end": interval_definition[1],
                    "body": body_sources,
                }
            )

        extents = gt_utils.flatten(node.compute_extent)
        functor_content = {
            "args": args,
            "regions": regions,
            "extents": extents,
        }

        return functor_content

    def visit_MultiStage(self, node: gt_ir.MultiStage) -> Dict[str, Any]:
        steps = [[stage.name for stage in group.stages] for group in node.groups]
        stage_functors = {
            stage.name: self.visit(stage)
            for stage in (stage for group in node.groups for stage in group.stages)
        }
        return {
            "exec": self.ITERATION_ORDER_TO_GT_ORDER[node.iteration_order],
            "steps": steps,
            "stage_functors": stage_functors,
        }

    def visit_Computation(self, node: Computation) -> Dict[str, Any]:
        multistages = [self.visit(multistage) for multistage in node.multistages]

        positional_axes = sorted(
            list({node.axis for node in gt_ir.filter_nodes_dfs(node, gt_ir.AxisIndex)})
        )

        stage_functors = {}
        for multistage in multistages:
            stage_functors.update(multistage["stage_functors"])

        return {
            "multistages": multistages,
            "api_fields": node.api_fields,
            "arg_fields": node.arg_fields,
            "parameters": node.parameters,
            "stage_functors": stage_functors,
            "positional_axes": positional_axes,
        }

    def _make_field_attributes(self, field_decl: gt_ir.FieldDecl) -> Dict[str, str]:
        return {
            "name": field_decl.name,
            "dtype": self._make_cpp_type(field_decl.data_type),
            "naxes": len(field_decl.axes),
            "axes": field_decl.axes,
            "selector": tuple(axis in field_decl.axes for axis in self.impl_node.domain.axes_names),
        }

    def _collect_field_arguments(self, gtstencil: GTStencil) -> List[Dict[str, str]]:
        api_fields = []
        arg_fields = []

        for name in [
            arg.name
            for arg in gtstencil.api_signature
            if arg.name in gtstencil.fields and arg.name not in gtstencil.unreferenced
        ]:
            field_decl = gtstencil.fields[name]
            field_attributes = self._make_field_attributes(field_decl)
            api_fields.append(field_attributes)

        api_names = {arg.name for arg in gtstencil.api_signature}
        # These are sorted so that they are in a stable order
        arg_names = sorted([name for name in gtstencil.fields if name not in api_names])

        for name in arg_names:
            field_decl = gtstencil.fields[name]
            field_attributes = self._make_field_attributes(field_decl)
            upper_indices = self.impl_node.fields_extents[name].upper_indices
            lower_indices = self.impl_node.fields_extents[name].lower_indices
            shape_adds = [upper - lower for upper, lower in zip(upper_indices, lower_indices)]
            shape = [f"domain[{i}] + {shape_adds[i]}" for i in range(len(shape_adds))]
            halo = [max(0, -x) for x in lower_indices]
            arg_fields.append({"shape": shape, "halo": halo, **field_attributes})

        return api_fields, arg_fields

    def visit_GTStencil(self, node: GTStencil) -> Dict[str, Dict[str, str]]:
        computations = [self.visit(computation) for computation in node.computations]

        # allocated_fields are both api and arg fields
        api_fields, arg_fields = self._collect_field_arguments(node)

        # All temporaries become gt::tmp_arg at the top level
        tmp_fields = OrderedDict()
        for computation in node.computations:
            for name in sorted(computation.tmp_fields.keys()):
                field_decl = computation.tmp_fields[name]
                dtype = self._make_cpp_type(field_decl.data_type)
                if name not in tmp_fields:
                    tmp_fields[name] = {"name": name, "dtype": dtype}
                else:
                    existing_dtype = tmp_fields[name]["dtype"]
                    if dtype != existing_dtype:
                        raise TypeError(
                            f"Temporary {name} has conflicting dtypes: {existing_dtype} and {dtype}"
                        )

        parameters = [
            {"name": parameter.name, "dtype": self._make_cpp_type(parameter.data_type)}
            for name, parameter in node.parameters.items()
            if name not in node.unreferenced
        ]

        offset_limit = _extract_max_k_offset(node)
        k_axis = {"n_intervals": 1, "offset_limit": offset_limit}

        max_extent = functools.reduce(
            lambda a, b: a | b, node.fields_extents.values(), gt_definitions.Extent.zeros()
        )
        halo_sizes = tuple(max(lower, upper) for lower, upper in max_extent.to_boundary())

        constants = {}
        if node.externals:
            for name, value in node.externals.items():
                value = self._make_cpp_value(name)
                if value is not None:
                    constants[name] = value

        positional_axes = sorted(
            list({node.axis for node in gt_ir.filter_nodes_dfs(node, gt_ir.AxisIndex)})
        )

        stage_functors = {}
        for computation in computations:
            stage_functors.update(computation["stage_functors"])

        template_args = dict(
            api_fields=api_fields,
            arg_fields=arg_fields,
            tmp_fields=tuple(tmp_fields.values()),
            halo_sizes=halo_sizes,
            constants=constants,
            stage_functors=stage_functors,
            gt_backend=self.gt_backend_t,
            parameters=parameters,
            k_axis=k_axis,
            module_name=self.module_name,
            computations=computations,
            positional_axes=positional_axes,
            stencil_unique_name=self.class_name,
        )

        sources: Dict[str, Dict[str, str]] = {"computation": {}, "bindings": {}}
        for key, template in self.templates.items():
            if key in self.COMPUTATION_FILES:
                sources["computation"][key] = template.render(**template_args)
            elif key in self.BINDINGS_FILES:
                sources["bindings"][key] = template.render(**template_args)

        return sources


class GTPyModuleGenerator(gt_backend.PyExtModuleGenerator):
    def generate_imports(self) -> str:
        return (
            """
from gt4py import storage as gt_storage
        """
            + super().generate_imports()
        )

    def generate_implementation(self) -> str:
        gtstencil = impl_to_gtstencil(self.builder.implementation_ir)
        sources = gt_utils.text.TextBlock(indent_size=self.TEMPLATE_INDENT_SIZE)

        api_field_names = {
            arg.name for arg in gtstencil.api_signature if arg.name in gtstencil.fields
        }

        api_field_args = []
        parameter_args = []
        # Assumption used below that parameteters always follow fields in the API signature,
        # because GridTools asserts this.
        for arg in gtstencil.api_signature:
            if arg.name not in self.args_data["unreferenced"]:
                if arg.name in api_field_names:
                    api_field_args.append(arg.name)
                    api_field_args.append("list(_origin_['{}'])".format(arg.name))
                else:
                    parameter_args.append(arg.name)

        # Field args must precede parameter args
        args = api_field_args + parameter_args

        # only generate implementation if any multi_stages are present. e.g. if no statement in the
        # stencil has any effect on the API fields, this may not be the case since they could be
        # pruned.
        if self.builder.implementation_ir.has_effect:
            source = """
# Load or generate a GTComputation object for the current domain size
pyext_module.run_computation(list(_domain_), {run_args}, exec_info)
""".format(
                run_args=", ".join(args),
            )
            sources.extend(source.splitlines())
        else:
            sources.append("")

        return sources.text


class BaseGTBackend(gt_backend.BasePyExtBackend, gt_backend.CLIBackendMixin):

    GT_BACKEND_OPTS = {
        "add_profile_info": {"versioning": True, "type": bool},
        "clean": {"versioning": False, "type": bool},
        "debug_mode": {"versioning": True, "type": bool},
        "verbose": {"versioning": False, "type": bool},
    }

    GT_BACKEND_T: str

    MODULE_GENERATOR_CLASS = GTPyModuleGenerator

    PYEXT_GENERATOR_CLASS = GTPyExtGenerator

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)

        implementation_ir = self.builder.implementation_ir

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt_src_manager.has_gt_sources() and not gt_src_manager.install_gt_sources():
            raise RuntimeError("Missing GridTools sources.")

        pyext_module_name: Optional[str]
        pyext_file_path: Optional[str]
        if implementation_ir.has_effect:
            pyext_module_name, pyext_file_path = self.generate_extension()
        else:
            # if computation has no effect, there is no need to create an extension
            pyext_module_name, pyext_file_path = None, None

        # Generate and return the Python wrapper class
        return self.make_module(
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
        )

    def generate_computation(self, *, ir: Any = None) -> Dict[str, Union[str, Dict]]:
        if not ir:
            ir = self.builder.implementation_ir
        dir_name = f"{self.builder.options.name}_src"
        src_files = self.make_extension_sources(ir=ir)
        return {dir_name: src_files["computation"]}

    def generate_bindings(
        self, language_name: str, *, ir: Any = None
    ) -> Dict[str, Union[str, Dict]]:
        if not ir:
            ir = self.builder.implementation_ir
        if language_name != "python":
            return super().generate_bindings(language_name)
        dir_name = f"{self.builder.options.name}_src"
        src_files = self.make_extension_sources(ir=ir)
        return {dir_name: src_files["bindings"]}

    @abc.abstractmethod
    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        """
        Generate and build a python extension for the stencil computation.

        Returns the name and file path (as string) of the compiled extension ".so" module.
        """
        pass

    def make_extension(
        self, *, gt_version: int = 1, ir: Any = None, uses_cuda: bool = False
    ) -> Tuple[str, str]:
        if not ir:
            # in the GTC backend, `ir` is the definition_ir
            ir = self.builder.implementation_ir
        # Generate source
        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            gt_pyext_sources: Dict[str, Any] = self.make_extension_sources(ir=ir)
            gt_pyext_sources = {**gt_pyext_sources["computation"], **gt_pyext_sources["bindings"]}
        else:
            # Pass NOTHING to the self.builder means try to reuse the source code files
            gt_pyext_sources = {
                key: gt_utils.NOTHING for key in self.PYEXT_GENERATOR_CLASS.TEMPLATE_FILES.keys()
            }

        # Build extension module
        pyext_opts = dict(
            verbose=self.builder.options.backend_opts.get("verbose", False),
            clean=self.builder.options.backend_opts.get("clean", False),
            **pyext_builder.get_gt_pyext_build_opts(
                debug_mode=self.builder.options.backend_opts.get("debug_mode", False),
                add_profile_info=self.builder.options.backend_opts.get("add_profile_info", False),
                uses_cuda=uses_cuda,
                gt_version=gt_version,
            ),
        )

        result = self.build_extension_module(gt_pyext_sources, pyext_opts, uses_cuda=uses_cuda)
        return result

    def make_extension_sources(self, *, ir) -> Dict[str, Dict[str, str]]:
        """Generate the source for the stencil independently from use case."""
        if "computation_src" in self.builder.backend_data:
            return self.builder.backend_data["computation_src"]
        class_name = self.pyext_class_name if self.builder.stencil_id else self.builder.options.name
        module_name = (
            self.pyext_module_name
            if self.builder.stencil_id
            else f"{self.builder.options.name}_pyext"
        )
        gt_pyext_generator = self.PYEXT_GENERATOR_CLASS(
            class_name, module_name, self.GT_BACKEND_T, self.builder.options
        )
        gt_pyext_sources = gt_pyext_generator(ir)
        final_ext = ".cu" if self.languages and self.languages["computation"] == "cuda" else ".cpp"
        comp_src = gt_pyext_sources["computation"]
        for key in [k for k in comp_src.keys() if k.endswith(".src")]:
            comp_src[key.replace(".src", final_ext)] = comp_src.pop(key)
        self.builder.backend_data["computation_src"] = gt_pyext_sources
        return gt_pyext_sources


@gt_backend.register
class GTX86Backend(BaseGTBackend):

    GT_BACKEND_T = "x86"

    name = "gtx86"
    options = BaseGTBackend.GT_BACKEND_OPTS
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": make_x86_layout_map,
        "is_compatible_layout": x86_is_compatible_layout,
        "is_compatible_type": gtcpu_is_compatible_type,
    }

    languages = {"computation": "c++", "bindings": ["python"]}

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=False)


@gt_backend.register
class GTMCBackend(BaseGTBackend):

    GT_BACKEND_T = "mc"

    name = "gtmc"
    options = BaseGTBackend.GT_BACKEND_OPTS
    storage_info = {
        "alignment": 8,
        "device": "cpu",
        "layout_map": make_mc_layout_map,
        "is_compatible_layout": mc_is_compatible_layout,
        "is_compatible_type": gtcpu_is_compatible_type,
    }

    languages = {"computation": "c++", "bindings": ["python"]}

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=False)


class GTCUDAPyModuleGenerator(GTPyModuleGenerator):
    def generate_implementation(self) -> str:
        source = (
            super().generate_implementation()
            + """
cupy.cuda.Device(0).synchronize()
    """
        )
        return source

    def generate_imports(self) -> str:
        source = (
            """
import cupy
"""
            + super().generate_imports()
        )
        return source

    def generate_pre_run(self) -> str:
        field_names = [
            key
            for key in self.args_data["field_info"]
            if self.args_data["field_info"][key] is not None
        ]

        return "\n".join([f + ".host_to_device()" for f in field_names])

    def generate_post_run(self) -> str:
        output_field_names = [
            name
            for name, info in self.args_data["field_info"].items()
            if info is not None and info.access == gt_definitions.AccessKind.READ_WRITE
        ]

        return "\n".join([f + "._set_device_modified()" for f in output_field_names])


@gt_backend.register
class GTCUDABackend(BaseGTBackend):

    MODULE_GENERATOR_CLASS = GTCUDAPyModuleGenerator

    GT_BACKEND_T = "cuda"

    name = "gtcuda"
    options = BaseGTBackend.GT_BACKEND_OPTS
    storage_info = {
        "alignment": 32,
        "device": "gpu",
        "layout_map": cuda_layout,
        "is_compatible_layout": cuda_is_compatible_layout,
        "is_compatible_type": cuda_is_compatible_type,
    }

    languages = {"computation": "cuda", "bindings": ["python"]}

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=True)
