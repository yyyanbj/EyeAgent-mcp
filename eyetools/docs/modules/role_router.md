# Role Router (RoleRouter)

Filter order: include -> exclude -> tags_any -> tags_all.
select_mode=manual returns candidate list requiring external selection.

Example config:
```yaml
roles:
  doctor:
    include: ["vision.*"]
    exclude: ["*.experimental"]
    tags_any: ["vision"]
    tags_all: ["stable"]
    select_mode: auto
```
