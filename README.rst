=========================================================
Joint operation of verification and anti-spoofing systems
=========================================================

This package contains methods for fusion of verification algorithms with anti-spoofing methods at decision and score-level, as well as well as for evaluation of the fused systems under spoofing attacks. In particular, the scripts in this package enable fusion of face verification and anti-spoofing algorithms on the `Replay-Attack <https://www.idiap.ch/dataset/replayattack>`_ face spoofing database. The fusion scripts require score files for both the verification and anti-spoofing algorithm. Hence, at least two score files are needed for each video in Replay-Attack: one for the verification and one for the anti-spoofing scores. Each score file contains the scores for all the frames in the corresponding video of Replay-Attack. If there is no score for a particular frame, the score value needs to be Nan. The format of the score files is .hdf5. The score files should be organized in the same directory structure as the videos in Replay-Attack.

Some of the scripts can receive multiple score files as input, enabling fusion of more then one verification algorithm with more then one anti-spoofing algorithm. 

To summarize, the methods in this package enable the user to:
  - parse score files in 4 column or 5 column format (`format <http://www.idiap.ch/software/bob/docs/releases/last/sphinx/html/measure/index.html?highlight=four#bob.measure.load.split_four_column>`_ specified in `Bob <http://www.idiap.ch/software/bob>`_) and extract the necessary information from that to organize the score files for each video into Replay-Attack directory structure.
  - evaluate the threshold of a classification system on the development set
  - apply the threshold on an evaluation or any other set
  - do: AND, SUM, LR and PLR fusion of verification and anti-spoofing system(s)
  - plot performance curves
  - plot score distributions
  - plot scatter plots and decision boundaries

If you use this package and/or its results, please cite the following
publications:
 
1. `Bob <http://www.idiap.ch/software/bob>`_ as the core framework used to run the experiments::

    @inproceedings{Anjos_ACMMM_2012,
        author = {A. Anjos AND L. El Shafey AND R. Wallace AND M. G\"unther AND C. McCool AND S. Marcel},
        title = {Bob: a free signal processing and machine learning toolbox for researchers},
        year = {2012},
        month = oct,
        booktitle = {20th ACM Conference on Multimedia Systems (ACMMM), Nara, Japan},
        publisher = {ACM Press},
    }

2. `Anti-spoofing in action: joint operation with an verification system <http://publications.idiap.ch/index.php/publications/show/2573>`_ ::
    
    @INPROCEEDINGS{Chingovska_CVPRWORKSHOPONBIOMETRICS_2013,
         author = {Chingovska, Ivana and Anjos, Andr{\'{e}} and Marcel, S{\'{e}}bastien},
       keywords = {biometric recognition, Counter-Measures, Fusion, Spoofing, trustworthy, vulnerability},
       projects = {Idiap, TABULA RASA, BEAT},
          month = jun,
          title = {Anti-spoofing in action: joint operation with a verification system},
      booktitle = {Proceedings of CVPR 2013},
           year = {2013},
       location = {Portland, Oregon},
    }

Installation
------------

.. note:: 

  If you are reading this page through our GitHub portal and not through PyPI,
  note **the development tip of the package may not be stable** or become
  unstable in a matter of moments.

  Go to `http://pypi.python.org/pypi/antispoofing.fusion_faceverif
  <http://pypi.python.org/pypi/antispoofing.fusion_faceverif>`_ to download the latest
  stable version of this package. Then, extract the .zip file to a folder of your choice.

The ``antispoofing.fusion_faceverif`` package is a satellite package of the free signal processing and machine learning library Bob_. This dependency has to be downloaded manually. This version of the package depends on Bob_ version 2 or greater. To install `packages of Bob <https://github.com/idiap/bob/wiki/Packages>`_, please read the `Installation Instructions <https://github.com/idiap/bob/wiki/Installation>`_. For Bob_ to be able to work properly, some dependent Bob packages are required to be installed. Please make sure that you have read the Dependencies for your operating system.

The most simple solution is to download and extract ``antispoofing.fusion_faceverif`` package, then to go to the console and write::

  $ cd antispoofing.fusion_faceverif
  $ python bootstrap-buildout.py
  $ bin/buildout

This will download all required dependent Bob_ and other packages and install them locally. 

Requirements and dependencies
-----------------------------

As mentioned before, this satellite package requires verification and anti-spoofing score files. To generate the face verification scores for the results in the paper `Anti-spoofing in action: joint operation with a verification system <http://publications.idiap.ch/index.php/publications/show/2573>`_, we used `FaceRecLib <https://github.com/bioidiap/facereclib>`_. To generate the anti-spoofing scores, we used the anti-spoofing algorithm from the following satelite packages: `antispoofing.lbptop <https://pypi.python.org/pypi/antispoofing.lbptop>`_ for LBP and LBP-TOP counter-measures and `antispoofing.motion <https://pypi.python.org/pypi/antispoofing.motion>`_ for motion-based counter-measure. Both the face verification and anti-spoofing scores were generated on per-frame basis. Of course, you can experiment with different verification and anti-spoofing algorithms, as long as your score files are organized as the directory structure of Replay-Attack.

This satellite package relies on the following satellite packages: `antispoofing.utils <https://pypi.python.org/pypi/antispoofing.utils>`_ and `antispoofing.fusion <https://pypi.python.org/pypi/antispoofing.fusion>`_. 

User guide
----------

This section explains the step by step procedure how to generate the results presented in the paper `Anti-spoofing in action: joint operation with a verification system <http://publications.idiap.ch/index.php/publications/show/2573>`_. The code is tied to `Replay-Attack <https://www.idiap.ch/dataset/replayattack>`_ database at the moment.

Step 1: Generate score files from the verification and anti-spoofing algorithms
===============================================================================

The first step is to train a face verification algorithm and to create models for each user into it. To generate the face verifications scores, you need to create a protocol for matching real-access (LICIT protocol) and spoof (SPOOF protocol) samples to user models that the algorithms has learned. The LICIT protocol is created by exhaustively matching each real access sample to the user model belonging to the sample's user and to all the other models. The SPOOF protocol is created my matching the spoof samples to the user model belonging to the sample's user. In our case, the algorithms work on a frame-by-frame basis. Due to computational limitations, we computed the scores only for every 10th frame of each video. The matching files for the LICIT and SPOOF protocol were then fed into FaceRecLib.

To generate the anti-spoofing scores, simply pick your favourite face verification algorithm and feed Replay-Attack to it. 

This user guide does not give details on the exact commands how to generate the scores. To learn how to do it, please refer directly to FaceRecLib and the anti-spoofing algorithm of your choice. 

Step 2: Convert the score files to the requested directory structure
====================================================================

As explained before, the score files need to be organized as the directory structure of Replay-Attack. While the anti-spoofing algorithms we use already give the scores in this format, FaceRecLib outputs score files in 4-column format (`format <http://www.idiap.ch/software/bob/docs/releases/last/sphinx/html/measure/index.html?highlight=four#bob.measure.load.split_four_column>`_), particularly, separate score files for the real accesses (LICIT protocol) and attacks (SPOOF protocol) videos. So, the first step is to convert them into the required format. For example, to convert the licit scores in the development set, run the following command::

    $ ./bin/four_column_to_dir_structure.py score_file out_dir -t licit -s devel replay 
    
The arguments ``score_file`` and ``out_dir`` refer to the 4-column score file which is input, and the directory for the converted scores, respectively. To see all the options for the script ``four_column_to_dir_structure.py``.
just type ``--help`` at the command line. If you want to do the conversion for a particular protocol of Replay-Attack (eg. print protocol), state that protocol at the end of the command::
 
    $ ./bin/four_column_to_dir_structure.py score_file out_dir -t licit -s devel replay print
    
Do not forget to do this step for all the dataset subsets (train, development and test set) and the two protocols (LICIT and SPOOF), using the appropriate input files and script options. Depending on the protocol, the scores will be saved into subdirectories called ``licit`` and ``spoof``  within ``out_dir``.
    
The score files in 4-column format generated by the recognition algorithm of FaceRecLib used in our work are supplied in this satellite package for your convenience. They can be found in the directory named ``supplemental_data``.

If it happens that your face verification or anti-spoofing algorithms output the scores in different format, feel free to implement your own convertor to get the scores into Replay-Attack directory structure.

Step 3: Decision-level fusion
=============================

AND decision fusion is supported via the script ``and_decision_fusion.py``. AND decision fusion depends on the decision thresholds of the verification and anti-spoofing algorithms separately. Therefore, we first need to compute them::

    $ ./bin/antispoof_threshold.py as_score_dir replay
    $ ./bin/faceverif_threshold.py fv_score_dir replay
    
The arguments as_score_dir and fv_score_dir refer to the directory with the score files for the anti-spoofing and face verification threshold respectively. The thresholds calculated with these methods are then fed as an input to the ``and_decision_fusion.py`` script::

    $ ./bin/and_decision_fusion.py -s fv_score_dir -a as_score_dir --ft fv_thr --at as_thr -v replay
    
The script directly prints the error rates. To see all the options for the script ``and_decision_fusion.py``
just type ``--help`` at the command line.

Step 4: Score-level fusion
==========================    

Three strategies for score-level fusion are available: SUM, LR and PLR. The score-fusion can be performed using the script ``fusion_fvas.py``::

    $ ./bin/fusion_fvas.py -s fv_score_dir -a as_score_dir -o outdir
    
The script writes the fused scores for each file in the specified output directory in a 4-column format. Having them, you can easily run any script for computing the performance or plotting. Note that you need to run this script separately for the LICIT and the SPOOF protocol for both development and test set at least. This will result in a total of 4 score files. To see all the options for the script ``fusion_fvas.py``, just type ``--help`` at the command line. A very important parameter is ``--sp`` that will save the normalization parameters and the machine of the fusion for further use.

Step 5: Compute performance
===========================

To compute the performance using the 4-column format score-files containing the fused scores, you can use the scripts ``eval_threshold.py`` to calculate the threshold on the development set and ``apply_threshold.py`` to compute the performance using the obtained treshold. Do not forget that you have 4 score files (one for development and one for test set for the LICIT and SPOOF protocol), and depending on your needs, you can use any of them for evaluation or application of the threshold.::

    $ ./bin/eval_threshold.py -s develset_score_file -c eer
    $ ./bin/apply_threshold.py -s testset_score_file -t thr

Usually, the development set score file of the LICIT protocol is used to evaluate the threshold. That threshold can be used on any of the 4 score files.

To see all the options for the scripts, just type ``--help`` at the command line.

Step 6: Plot performance curves
===============================

Using the script ``plot_on_demand.py``, you can choose to plot many different plots like score distributions or DET curves on the LICIT, SPOOF protocol or both. Just see at the documentation of the script to see what input you need to specify for the desired curve. As mandatory input, you need to give the score files for the LICIT and SPOOF protocol for both the development and test set.::

    $ ./bin/plot_on_demand.py devel_licit_scores eval_licit_scores devel_spoof_scores eval_spoof_scores -b eer -i 2
    
This will plot the DET curve of the LICIT protocol overlayed with the DET curve of the SPOOF protocol. To see all the options for the script ``plot_on_demand.py``, just type ``--help`` at the command line.

Step 7: Scatter plots
=====================

Scatter plots plot the verification and anti-spoofing scores in the 2D space, together with a decision boundary depending on the algorithms used for their fusion. To plot a scatter plot for LLR fused scores, type::

    $ ./bin/scatter_plot.py -s fv_score_dir -a as_score_dir -m machine_file -n norm_file -d thr -f LLR
    
The devel threshold (specified with ``-d`` parameter) is a mandatory argument in this script. In the case of AND fusion (option ``-f AND``), two thresholds need to be specified with this argument. Normalization parameter (parameter ``-n norm_file``) needs to be specified for the score fusion algorithms (i.s. option ``-f SUM``, ``-f LLR`` or ``-f LLR_P``), where norm_file is a file containing the normalization parameters. Usually, this is the file saved when the option ``--sp`` is set when running the script ``fvas_fusion.py`` in Step 4. Similarly, the score fusion algorithms require the parameter ``-m machine_file``, where machine_file contains the of the fusion algorithm. It is also saved when the option ``--sp`` is set when running the script ``fvas_fusion.py`` in Step 4.
 
To see all the options for the script ``scatter_plot.py``, just type ``--help`` at the command line. 

Problems
--------

In case of problems, please contact ivana.chingovska@idiap.ch
