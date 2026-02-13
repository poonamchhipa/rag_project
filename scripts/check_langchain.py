import pkgutil, importlib
mods = [m.name for m in pkgutil.iter_modules() if 'langchain' in m.name]
print('langchain-related modules:', mods)
try:
    m = importlib.import_module('langchain_community')
    print('langchain_community importable at', getattr(m, '__file__', m))
except Exception as e:
    print('langchain_community import error:', repr(e))
