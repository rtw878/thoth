.. Thoth documentation master file, created by
   sphinx-quickstart on Thu Jan 1 00:00:00 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../assets/new-logos/PRIMARY-LOCKUP-cropped.svg
   :alt: Thoth
   :align: center
   :width: 600px

Welcome to Thoth's documentation!
=================================

Thoth is an AI-powered application for transcribing historical handwriting
using state-of-the-art machine learning models. Built on the TrOCR architecture
and fine-tuned using Parameter-Efficient Fine-Tuning (LoRA), it provides accurate
transcriptions while being computationally efficient.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide
   api_reference


Overview
--------

Thoth is designed to help historians, archivists, and digital humanities
researchers transcribe historical documents with challenging handwriting. The project
combines modern machine learning techniques with a user-friendly interface to make
historical document transcription accessible to non-technical users.

Key Features
^^^^^^^^^^^^

- **State-of-the-art HTR**: Powered by TrOCR (Transformer-based Optical Character Recognition)
- **Multi-language support**: Fine-tuned models for various historical scripts
- **User-friendly GUI**: Cross-platform desktop application built with PyQt6
- **Parameter-efficient training**: Uses LoRA for efficient fine-tuning
- **Comprehensive preprocessing**: Advanced image processing pipeline for historical documents

Architecture
^^^^^^^^^^^^

The system is built on several key components:

1. **Data Pipeline**: Automated downloading and preprocessing of historical datasets
2. **Model Training**: Fine-tuning TrOCR models using LoRA for efficiency
3. **GUI Application**: PyQt6-based desktop application for end users
4. **Evaluation Framework**: Comprehensive metrics including CER and WER

Quick Start
-----------

For new users, we recommend starting with the :doc:`installation` guide to set up
the development environment, then proceeding to the :doc:`user_guide` for instructions
on using the application.

For developers interested in extending the project, the :doc:`api_reference` provides
detailed documentation of the codebase.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
