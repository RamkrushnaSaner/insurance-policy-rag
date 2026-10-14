# Policy document

The notebook reads the policy as a plain UTF-8 text file. The document is not included in this
repository because it belongs to HDFC ERGO General Insurance Company Limited.

## Getting the text

1. Download the **my:Optima Secure prospectus** (PDF) from the
   [HDFC ERGO Optima Secure page](https://www.hdfcergo.com/health-insurance/optima-secure).
2. Convert the PDF to plain text (`.txt`, UTF-8) with a PDF-to-text tool.
3. Save the result in this folder as `policy.txt`.

Everything in this folder except this README is ignored by git, so the policy text is never
committed by accident.

## Using it

- **Colab:** upload the file when Section 1 of the notebook shows the file picker.
- **Local:** in Section 1, set `SOURCE_PATH = "data/policy.txt"`.

## Checking the conversion

Different PDF converters lay text out differently. After running Section 2.1, check that the
printed section ladder is a clean, increasing sequence of section numbers that ends with the
annexures. If a section you can see in the PDF is missing:

- a repeated page header may be splitting the heading — add it to `PAGE_HEADER_PATTERNS` in Section 1;
- or the heading is preceded by a word the parser treats as a cross-reference — see `REF_WORDS` in Section 2.

The gold questions and the regression fixture in Section 10 were built on the author's text export.
A different export can shift chunk boundaries slightly, so expect small differences in those scores.
