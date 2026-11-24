#!/bin/sh
# Install the repository's canonical skill, with an optional isolated destination.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SOURCE="$SCRIPT_DIR/../skills/switchit/SKILL.md"
TARGET_DIR="${HOME}/.agents/skills/switchit"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dest)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "error: --dest requires a directory" >&2
                exit 2
            fi
            case "$2" in
                -*) echo "error: --dest requires a directory, not an option" >&2; exit 2 ;;
            esac
            TARGET_DIR=$2
            shift 2
            ;;
        --help|-h)
            echo 'Usage: install-skill.sh [--dest DIRECTORY]'
            echo 'Default: ~/.agents/skills/switchit. Installs SKILL.md only.'
            exit 0
            ;;
        *) echo "error: unknown option $1" >&2; exit 2 ;;
    esac
done
if [ ! -f "$SOURCE" ]; then
    echo "error: skill not found at $SOURCE" >&2
    exit 1
fi
mkdir -p "$TARGET_DIR"
if [ -f "$TARGET_DIR/SKILL.md" ] && cmp -s "$SOURCE" "$TARGET_DIR/SKILL.md"; then
    echo "The switchit skill is already up to date at $TARGET_DIR/SKILL.md"
    exit 0
fi
TEMPORARY=$(mktemp "$TARGET_DIR/.SKILL.md.XXXXXX")
trap 'rm -f "$TEMPORARY"' EXIT HUP INT TERM
cp "$SOURCE" "$TEMPORARY"
chmod 644 "$TEMPORARY"
mv -f "$TEMPORARY" "$TARGET_DIR/SKILL.md"
trap - EXIT HUP INT TERM
echo "Installed the switchit skill to $TARGET_DIR/SKILL.md"
echo 'Invoke $switchit from a compatible agent. The Python harness is installed separately.'
