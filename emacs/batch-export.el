;;; batch-export.el --- Batch exporter for Prolog RLM Lab -*- lexical-binding: t; -*-
(let* ((this (or load-file-name buffer-file-name))
       (dir (file-name-directory this)))
  (load-file (expand-file-name "rlm-lab.el" dir)))
(rlm-lab-export-report)
