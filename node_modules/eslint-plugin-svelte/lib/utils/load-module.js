import Module from 'module';
import path from 'path';
const cache = new WeakMap();
const cache4b = new Map();
/**
 * Load module
 */
export function loadModule(context, name) {
    const key = context.sourceCode.ast;
    let modules = cache.get(key);
    if (!modules) {
        modules = {};
        cache.set(key, modules);
    }
    const mod = modules[name] || cache4b.get(name);
    if (mod)
        return mod;
    try {
        // load from cwd
        const cwd = context.cwd ?? process.cwd();
        const relativeTo = path.join(cwd, '__placeholder__.js');
        return (modules[name] = Module.createRequire(relativeTo)(name));
    }
    catch {
        // ignore
    }
    for (const relativeTo of [
        // load from lint file name
        context.filename,
        // load from lint file name (physical)
        context.physicalFilename,
        // load from this plugin module
        typeof __filename !== 'undefined' ? __filename : ''
    ]) {
        if (relativeTo) {
            try {
                return (modules[name] = Module.createRequire(relativeTo)(name));
            }
            catch {
                // ignore
            }
        }
    }
    return null;
}
