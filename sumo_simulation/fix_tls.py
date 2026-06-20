import re
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
net_path = os.path.join(script_dir, 'osm_cut.net.xml')

with open(net_path, 'r', encoding='utf-8') as f:
    content = f.read()

# New phase order: leading protected left turn FIRST, then through
# Phase 0 (8s):  EW left protected  - state rrrGGrrrrrrrGGrrrr
# Phase 1 (3s):  yellow EW left     - state rrryyrrrrrrryyrrrr
# Phase 2 (30s): EW through, left=r - state GGGrrrrrrGGGrrrrrr
# Phase 3 (3s):  yellow EW through  - state yyyrrrrrryyyrrrrrr
# Phase 4 (8s):  NS left protected  - state rrrrrrrrGrrrrrrrrG
# Phase 5 (3s):  yellow NS left     - state rrrrrrrryrrrrrrrry
# Phase 6 (30s): NS through, left=r - state rrrrrGGGrrrrrrGGGr
# Phase 7 (3s):  yellow NS through  - state rrrrryyyrrrrrryyyr

new_tl = (
    '    <tlLogic id="cluster_53190763_5896114911" type="static" programID="0" offset="0">\n'
    '        <!-- Phase 0 (30s): Eastbound Green (indices 0, 1, 2, 3) -->\n'
    '        <phase duration="30" state="GGGGrrrrrrrrrr"/>\n'
    '        <!-- Phase 1 (3s): Eastbound Yellow -->\n'
    '        <phase duration="3"  state="yyyyrrrrrrrrrr"/>\n'
    '        <!-- Phase 2 (30s): Northbound Green (indices 4, 5, 6) -->\n'
    '        <phase duration="30" state="rrrrGGGrrrrrrr"/>\n'
    '        <!-- Phase 3 (3s): Northbound Yellow -->\n'
    '        <phase duration="3"  state="rrrryyyrrrrrrr"/>\n'
    '        <!-- Phase 4 (30s): Westbound Green (indices 7, 8, 9, 10) -->\n'
    '        <phase duration="30" state="rrrrrrrGGGGrrr"/>\n'
    '        <!-- Phase 5 (3s): Westbound Yellow -->\n'
    '        <phase duration="3"  state="rrrrrrryyyyrrr"/>\n'
    '        <!-- Phase 6 (30s): Southbound Green (indices 11, 12, 13) -->\n'
    '        <phase duration="30" state="rrrrrrrrrrrGGG"/>\n'
    '        <!-- Phase 7 (3s): Southbound Yellow -->\n'
    '        <phase duration="3"  state="rrrrrrrrrrryyy"/>\n'
    '    </tlLogic>'
)

pattern = r'    <tlLogic id="cluster_53190763_5896114911".*?</tlLogic>'
new_content = re.sub(pattern, new_tl, content, flags=re.DOTALL)

if new_content == content:
    print('ERROR: Pattern not found, no replacement made!')
else:
    with open(net_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('SUCCESS: tlLogic updated with leading protected left turns')

# Verify
with open(net_path, 'r', encoding='utf-8') as f:
    verify = f.read()
import re as re2
m = re2.search(r'<tlLogic.*?</tlLogic>', verify, re2.DOTALL)
if m:
    print('\nVerification - new tlLogic block:')
    print(m.group())
