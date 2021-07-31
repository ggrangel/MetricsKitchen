#  -*- coding: utf-8 -*-
"""

Author: Gustavo B. Rangel
Date: 19/07/2021

"""

import graphviz

from pathlib import Path

dot = graphviz.Digraph(filename=Path(__file__).parent / 'test', format='png')

dot.node('T', 'T', pos="0,0!")
dot.node('Y', 'Y')
dot.node('W', 'W')
dot.node('Z', 'Z')

dot.edges(['ZT', 'WT', 'TY', 'WY'])

print(dot.source)

dot.render(view=True)

