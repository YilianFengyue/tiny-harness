import subprocess
import sys

raise SystemExit(subprocess.call([sys.executable, '-m', 'pytest', '-q']))
