#!/bin/bash

set -euo pipefail

<<<<<<< HEAD
# if [ "${1:0:1}" = "-" ]; then
#     set -- deeply "$@"
# fi
=======
if [ "${1:0:1}" = "-" ]; then
    set -- deeply "$@"
fi
>>>>>>> template/master

exec "$@"