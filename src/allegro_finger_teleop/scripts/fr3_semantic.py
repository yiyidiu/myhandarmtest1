#!/usr/bin/env python3
"""SRDF for the FR3 chain and Allegro contact surfaces used by Gazebo."""
import itertools
import sys
from xml.etree import ElementTree as ET

model = ET.parse(sys.argv[1])
semantic = ET.Element("robot", name="fr3_allegro")
group = ET.SubElement(semantic, "group", name="fr3_arm")
ET.SubElement(group, "chain", base_link="fr3_link0", tip_link="tool0")
parent = {j.find("child").get("link"): j.find("parent").get("link") for j in model.findall("joint")}
bodies = {link.get("name") for link in model.findall("link") if link.find("collision") is not None}
for name in sorted(bodies):
    ancestor = parent.get(name)
    while ancestor and ancestor not in bodies:
        ancestor = parent.get(ancestor)
    if ancestor:
        ET.SubElement(semantic, "disable_collisions", link1=ancestor, link2=name, reason="Adjacent")
# Phalanges of different fingers can touch during opposition. Such
# contact is intentional in a closed fist, knuckle pinch or multi-pad pinch.
digits = {digit: tuple("algr_lh_" + digit + "_" + link for link in segments)
          for digit, segments in (("th", ("px", "ds")), ("if", ("px", "md", "ds")),
                                  ("mf", ("px", "md", "ds")), ("rf", ("px", "md", "ds")))}
for a, b in itertools.combinations(digits.values(), 2):
    for link_a, link_b in itertools.product(a, b):
        ET.SubElement(semantic, "disable_collisions", link1=link_a, link2=link_b, reason="GraspContact")
ET.SubElement(semantic, "disable_collisions", link1="algr_lh_palm", link2="algr_lh_th_bs", reason="JointHousing")
print(ET.tostring(semantic, encoding="unicode"))
