;;; rlm-lab.el --- Org/gptel helpers for Prolog RLM Lab -*- lexical-binding: t; -*-

(require 'cl-lib)
(require 'subr-x)
(require 'org)
(require 'ob)
(require 'ox-html)

(defconst rlm-lab-root
  (file-name-directory
   (directory-file-name
    (file-name-directory (or load-file-name buffer-file-name default-directory))))
  "Absolute project root.")

(defun rlm-lab-path (&rest parts)
  "Return a path below `rlm-lab-root' from PARTS."
  (let ((path rlm-lab-root))
    (dolist (part parts path)
      (setq path (expand-file-name part path)))))

(defun rlm-lab-read-file (path)
  "Read PATH as text, returning an empty string if it does not exist."
  (if (file-readable-p path)
      (with-temp-buffer
        (insert-file-contents path)
        (buffer-string))
    ""))

(defun rlm-lab-setup ()
  "Enable the Babel languages used by the lab and load ob-async when present."
  (interactive)
  (org-babel-do-load-languages
   'org-babel-load-languages
   '((emacs-lisp . t)
     (python . t)
     (shell . t)))
  (if (require 'ob-async nil t)
      (message "Prolog RLM Lab: ob-async enabled")
    (display-warning
     'rlm-lab
     "ob-async is not installed. Babel blocks still run, but :async headers will not be asynchronous."
     :warning))
  (setq-local org-confirm-babel-evaluate nil))

(defconst rlm-lab-findings-system
  "You are writing the findings section of a technical benchmark report. Use only the supplied benchmark evidence. Be concise, literal, and quantitative. Do not use hype, marketing language, anthropomorphic claims, vague praise, fake certainty, or filler. Do not claim that a metric measures hidden cognition when it measures observable runtime behavior. Distinguish synthetic/demo data from real benchmark data. If evidence is insufficient, state that directly. Return valid Org mode only, beginning with '* Findings'. Do not use Markdown fences.")

(defun rlm-lab-findings-template ()
  "Return the canonical FINDINGS.org template."
  (concat
   "* Findings\n\n"
   "#+begin_quote\n"
   "Pending generated analysis. Claims in this section must be supported by the benchmark summaries.\n"
   "#+end_quote\n\n"
   "** Capability substitution\n\nPending evidence.\n\n"
   "** Recursion behavior\n\nPending evidence.\n\n"
   "** Symbolic offload\n\nPending evidence.\n\n"
   "** Failure modes\n\nPending evidence.\n\n"
   "** Cost and latency\n\nPending evidence.\n\n"
   "** Evidence-backed conclusions\n\nPending evidence.\n\n"
   "** Caveats\n\nPending evidence.\n\n"
   "** Reproducibility\n\nPending evidence.\n"))

(defun rlm-lab-write-findings-template ()
  "Reset FINDINGS.org to the canonical template."
  (interactive)
  (with-temp-file (rlm-lab-path "FINDINGS.org")
    (insert (rlm-lab-findings-template))))

(defun rlm-lab-findings-context ()
  "Build the evidence packet supplied to gptel."
  (let ((metrics (rlm-lab-read-file (rlm-lab-path "results/summary/metrics.org")))
        (capability (rlm-lab-read-file (rlm-lab-path "results/summary/capability.json")))
        (summary (rlm-lab-read-file (rlm-lab-path "results/summary/summary.csv"))))
    (format
     (concat
      "Write FINDINGS.org for the Prolog-RLM capability-substitution experiment.\n\n"
      "Required headings:\n"
      "* Findings\n"
      "** Capability substitution\n"
      "** Recursion behavior\n"
      "** Symbolic offload\n"
      "** Failure modes\n"
      "** Cost and latency\n"
      "** Evidence-backed conclusions\n"
      "** Caveats\n"
      "** Reproducibility\n\n"
      "Interpret CRR only when established. Treat Symbolic Work Share as the share of observable model/Prolog calls, not a percentage of consciousness or hidden thought. Compare recursion levels, task families, token use, latency, and failures. Prefer exact values over adjectives.\n\n"
      "--- METRICS.ORG ---\n%s\n"
      "--- CAPABILITY.JSON ---\n%s\n"
      "--- SUMMARY.CSV ---\n%s\n")
     metrics
     capability
     (if (> (length summary) 120000)
         (concat (substring summary 0 120000) "\n[summary truncated by lab context builder]\n")
       summary))))

(defun rlm-lab-findings-context-preview (&optional max-chars)
  "Return a preview of the evidence packet limited to MAX-CHARS."
  (let* ((max-chars (or max-chars 5000))
         (text (rlm-lab-findings-context)))
    (if (> (length text) max-chars)
        (concat (substring text 0 max-chars) "\n...[preview truncated]...")
      text)))

(defun rlm-lab-clean-gptel-org (response)
  "Normalize RESPONSE into a FINDINGS.org document."
  (let ((text (string-trim response)))
    (setq text (replace-regexp-in-string "\\````\\(?:org\\)?[[:space:]\n]*" "" text))
    (setq text (replace-regexp-in-string "[[:space:]\n]*```\\'" "" text))
    (unless (string-prefix-p "* Findings" text)
      (setq text (concat "* Findings\n\n" text)))
    (concat text "\n")))

(defun rlm-lab--write-gptel-findings (response info &optional export-after)
  "gptel callback: write RESPONSE using INFO; optionally EXPORT-AFTER."
  (if (stringp response)
      (progn
        (with-temp-file (rlm-lab-path "FINDINGS.org")
          (insert (rlm-lab-clean-gptel-org response)))
        (message "Prolog RLM Lab: FINDINGS.org generated with %s"
                 (or (plist-get info :model) "active gptel model"))
        (when export-after
          (rlm-lab-export-report)))
    (display-warning
     'rlm-lab
     (format "gptel findings generation failed: %s" (or (plist-get info :status) response))
     :error)))

(defun rlm-lab-generate-findings ()
  "Generate FINDINGS.org asynchronously with the active gptel backend/model."
  (interactive)
  (require 'gptel-request)
  (gptel-request
   (rlm-lab-findings-context)
   :system rlm-lab-findings-system
   :stream nil
   :callback (lambda (response info)
               (rlm-lab--write-gptel-findings response info nil))))

(defun rlm-lab-generate-findings-and-export ()
  "Generate findings asynchronously and export REPORT.org when complete."
  (interactive)
  (require 'gptel-request)
  (gptel-request
   (rlm-lab-findings-context)
   :system rlm-lab-findings-system
   :stream nil
   :callback (lambda (response info)
               (rlm-lab--write-gptel-findings response info t))))

(defun rlm-lab-export-report ()
  "Export REPORT.org to public/index.html with inlined CSS and JavaScript."
  (interactive)
  (let* ((default-directory rlm-lab-root)
         (report (rlm-lab-path "REPORT.org"))
         (output (rlm-lab-path "public/index.html"))
         (css (rlm-lab-read-file (rlm-lab-path "assets/report.css")))
         (js (rlm-lab-read-file (rlm-lab-path "assets/report.js")))
         (org-export-with-broken-links t)
         (org-html-validation-link nil)
         (org-html-preamble nil)
         (org-html-postamble nil)
         (org-html-head-include-default-style nil)
         (org-html-head-include-scripts nil)
         (org-html-head-extra
          (concat "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
                  "<style>\n" css "\n</style>\n"
                  "<script>\n" js "\n</script>\n")))
    (make-directory (file-name-directory output) t)
    (let ((plot-source (rlm-lab-path "results/plots"))
          (plot-target (rlm-lab-path "public/results/plots")))
      (when (file-directory-p plot-source)
        (when (file-directory-p plot-target)
          (delete-directory plot-target t))
        (copy-directory plot-source plot-target nil t t)))
    (with-current-buffer (find-file-noselect report)
      (org-export-to-file 'html output nil nil nil nil nil))
    (message "Prolog RLM Lab: exported %s" output)
    output))

(provide 'rlm-lab)
;;; rlm-lab.el ends here
