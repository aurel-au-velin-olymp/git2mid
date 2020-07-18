class config:
    """Get/set defaults for the :mod:`git2mid` module.

    For example, when you want to change the default cache folder::

        import git2mid
        git2mid.config.GIT2MID_CACHE_ROOT = '~/data'

    """

    GIT2MID_DB_ROOT = '~/data'
    """Cache folder."""

    GIT2MID_CHORD_ROOT = '~/data'
    """Cache folder."""
