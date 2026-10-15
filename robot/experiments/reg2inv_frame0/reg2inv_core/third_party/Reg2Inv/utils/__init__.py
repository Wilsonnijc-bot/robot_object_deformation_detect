"""Pin the upstream utility directory as a package in the shared-base environment.

Packaging compatibility only: avoid an unrelated installed ``utils`` package
shadowing the upstream modules. Algorithm modules are unchanged.
"""
