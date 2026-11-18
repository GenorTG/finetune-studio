'use strict';
// No shell interpolation, dependencies, credentials or protocol transformations.
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const [folder, keyfile] = process.argv.slice(2);
function fail(message) {
  console.error('Smart Brain: ' + message);
  process.exit(1);
}
if (process.platform !== 'linux') fail('this bridge supports Linux only.');
if (!folder || !keyfile || !path.isAbsolute(folder) || !path.isAbsolute(keyfile) ||
    /[\r\n\0]/.test(folder + keyfile)) fail('select absolute package and passphrase-file paths.');
try {
  fs.accessSync(path.join(folder, '.venv/bin/python'), fs.constants.X_OK);
  fs.accessSync(path.join(folder, 'run-mcp.sh'), fs.constants.R_OK);
  fs.accessSync(keyfile, fs.constants.R_OK);
} catch (_) {
  fail('package or keyfile unavailable. Extract the package, run bash install.sh, and check the selected paths.');
}
const child = spawn('bash', [path.join(folder, 'run-mcp.sh'), '--keyfile', keyfile],
  {stdio: 'inherit', shell: false});
child.on('error', () => fail('could not launch bash; check your Linux installation.'));
for (const signal of ['SIGTERM', 'SIGINT']) process.on(signal, () => child.kill(signal));
child.on('exit', (code) => { process.exitCode = code === 0 ? 0 : 1; });
