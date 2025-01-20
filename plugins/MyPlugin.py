from mkdocs.config import base, config_options as c


class _ValidationOptions(base.Config):
    enabled = c.Type(bool, default=True)
    verbose = c.Type(bool, default=False)
    skip_checks = c.ListOfItems(c.Choice(('foo', 'bar', 'baz')), default=[])


class MyPluginConfig(base.Config):
    definition_file = c.File(exists=True)  # required
    checksum_file = c.Optional(c.File(exists=True))  # can be None but must exist if specified
    validation = c.SubConfig(_ValidationOptions)


class MyPlugin(mkdocs.plugins.BasePlugin[MyPluginConfig]):
    config_scheme = (
        ('foo', c.Type(str, default='a default value')),
        ('bar', c.Type(int, default=0)),
        ('baz', c.Type(bool, default=True))
    )
