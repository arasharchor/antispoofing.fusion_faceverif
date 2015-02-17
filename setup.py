#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
#  Mon 25 Jun 2012 13:21:06 CEST

from setuptools import setup, find_packages

packages = ['antispoofing'] + ['antispoofing.' + p for p in find_packages('antispoofing')]

# Define package version
version = open("version.txt").read().rstrip()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    name='antispoofing.fusion_faceverif',
    version=version,
    description='Decision and score-level fusion tools for joint operation of face verification and anti-spoofing system',
    url='http://github.com/bioidiap/antispoofing.fusion_faceverif',
    license='LICENSE.txt',
    author='Ivana Chingovska',
    author_email='Ivana Chingovska <ivana.chingovska@idiap.ch>',
    #long_description=open('doc/howto.rst').read(),

    # This line is required for any distutils based packaging.
    packages=find_packages(),

    install_requires=[
        "bob.db.base", #1.1.0
        "argparse", # better option parsing
        "bob.db.replay", # Replay-Attack database    >= 1.0.4
        "antispoofing.utils",  #Utils Package >= 1.1.3
        "antispoofing.fusion", # Fusion utilities
    ],

    entry_points={
      'console_scripts': [
        'and_decision_fusion.py = antispoofing.fusion_faceverif.script.and_decision_fusion:main',
        'fusion_fvas.py = antispoofing.fusion_faceverif.script.fusion_fvas:main',
        'antispoof_threshold.py = antispoofing.fusion_faceverif.script.antispoof_threshold:main',
        'faceverif_threshold.py = antispoofing.fusion_faceverif.script.faceverif_threshold:main',
        'four_column_to_dir_structure.py = antispoofing.fusion_faceverif.script.four_column_to_dir_structure:main',
        'scatter_plot.py = antispoofing.fusion_faceverif.script.scatter_plot:main',
        'plot_on_demand.py = antispoofing.fusion_faceverif.script.plot_on_demand:main', 
        'apply_threshold.py = bob.measure.script.apply_threshold:main',
        'eval_threshold.py = bob.measure.script.eval_threshold:main',
        ],
      },

)
