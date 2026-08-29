#!/bin/sh
# Resolve the runtime UID/GID through libnss-wrapper when they have no
# /etc/passwd or /etc/group entry, then exec the original command.
#
# Kubernetes/OpenShift may assign arbitrary non-root UIDs that do not exist in
# the image. NSS wrapper provides a dynamic identity without modifying
# /etc/passwd, requiring root, or weakening OpenShift compatibility.

set -eu

uid="$(id -u)"
gid="$(id -g)"

passwd_file="${TMPDIR:-/tmp}/passwd"
group_file="${TMPDIR:-/tmp}/group"

if ! getent passwd "$uid" >/dev/null 2>&1 || ! getent group "$gid" >/dev/null 2>&1; then
    cp /etc/passwd "$passwd_file"
    cp /etc/group "$group_file"

    if ! getent passwd "$uid" >/dev/null 2>&1; then
        printf 'speech-transcriber:x:%s:%s:Speech Transcriber:%s:/sbin/nologin\n' \
            "$uid" "$gid" "${HOME:-/cache/home}" >> "$passwd_file"
    fi

    if ! getent group "$gid" >/dev/null 2>&1; then
        printf 'speech-transcriber:x:%s:\n' "$gid" >> "$group_file"
    fi

    export NSS_WRAPPER_PASSWD="$passwd_file"
    export NSS_WRAPPER_GROUP="$group_file"

    libnss_wrapper="$(find /usr/lib -name 'libnss_wrapper.so' -print -quit 2>/dev/null)"
    if [ -z "$libnss_wrapper" ]; then
        echo "uid-entrypoint: libnss_wrapper.so not found" >&2
        exit 1
    fi

    export LD_PRELOAD="$libnss_wrapper${LD_PRELOAD:+:$LD_PRELOAD}"
fi

exec "$@"
