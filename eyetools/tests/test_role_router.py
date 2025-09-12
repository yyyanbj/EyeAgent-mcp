from eyetools.core.role_router import RoleRouter
from dataclasses import dataclass


@dataclass
class DummyMeta:
    id: str
    tags: list[str]


def test_role_router_basic_include_exclude():
    cfg = {
        "roles": {
            "r1": {"include": ["a.*"], "exclude": ["a.block*"], "tags_any": ["x"]}
        },
        "defaults": {"include": ["*"]},
    }
    rr = RoleRouter(cfg)
    metas = [
        DummyMeta("a.tool1", ["x"]),
        DummyMeta("a.blocked", ["x"]),
        DummyMeta("b.other", ["x"]),
    ]
    res = rr.filter_tools("r1", metas)
    assert not res.selection_required
    assert res.tool_ids == ["a.tool1"]


def test_role_router_tags_all_and_manual():
    cfg = {
        "roles": {
            "r2": {"include": ["*"], "tags_all": ["x", "y"], "select_mode": "manual"}
        }
    }
    rr = RoleRouter(cfg)
    metas = [
        DummyMeta("m1", ["x", "y"]),
        DummyMeta("m2", ["x"]),
    ]
    res = rr.filter_tools("r2", metas)
    assert res.selection_required
    assert res.tool_ids == ["m1"]
