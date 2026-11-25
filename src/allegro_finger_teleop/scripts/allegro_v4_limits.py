#!/usr/bin/env python3
"""Apply the position limits from the Wonik V4 hardware table (degree column)."""
import argparse
from math import radians
from pathlib import Path
import sys
from lxml import etree

# https://www.allegrohand.com/sub/tech/pop.php?idx=23
LONG_DEG = ((-32.65, 32.65), (-16.95, 97.97), (-15.69, 103.65), (-18.73, 98.43))
THUMB_DEG = ((20.62, 85.71), (-11.68, 72.36), (-16.55, 99.92), (-14.89, 103.22))


def hardware_limits():
    for finger in ("if", "mf", "rf", "th"):
        names = ("cmc", "axl", "mcp", "ipl") if finger == "th" else ("axl", "mcp", "pip", "dip")
        for joint, span in zip(names, THUMB_DEG if finger == "th" else LONG_DEG):
            yield "algr_lh_{}_{}".format(finger, joint), tuple(map(radians, span))


def apply_limits(xml):
    model = etree.fromstring(xml)
    for name, (lower, upper) in hardware_limits():
        limit, = model.xpath("./joint[@name=$name and @type='revolute']/limit", name=name)
        limit.set("lower", format(lower, ".12f"))
        limit.set("upper", format(upper, ".12f"))
    return etree.tostring(model, encoding="UTF-8", xml_declaration=True) + b"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--control-point", choices=("wrist", "palm"), default="wrist")
    parser.add_argument("--write", action="store_true", help="update the installed static URDF in place")
    args = parser.parse_args()
    if args.model.suffix == ".xacro":
        if args.write:
            parser.error("--write requires a static URDF")
        import xacro
        xml = xacro.process_file(str(args.model), mappings={"control_point": args.control_point}).toxml(encoding="utf-8")
    else:
        xml = args.model.read_bytes()
    updated = apply_limits(xml)
    if args.write:
        args.model.write_bytes(updated)
    else:
        sys.stdout.buffer.write(updated)
