# External Drive Assets

Six task families reference source documents, spreadsheets, presentations, or Drive folders from their `task.md` prompts. The original files live in the authors' Google Drive; to run these families you must host your own copies:

1. **Download** `knows-assets.zip` from this repository's GitHub Release page.
2. **Upload** each item to your own Google Drive, preserving the per-family folder structure inside the zip. Files exported as Office formats must be converted back to Google formats on upload (Drive: right-click → *Open with* → Google Docs/Sheets/Slides, or enable *Convert uploads* in Drive settings). Set sharing so your **agent account can view** them (and, where the task asks the agent to add files to a folder, **edit**).
3. **Substitute the URLs/IDs** listed below with your copies' URLs. Substitutions are one-time edits per checkout.

| Family | Instances | Assets | Where to substitute |
|---|---|---|---|
| `docs_1_formal_letter` | i1–i5 | 1 Drive folder per instance (letter source materials) | The folder URL in each `instance_N/task.md` |
| `docs_11_personal_recipe_ocr` | i1–i5 | i1: 1 file (recipe scan); i2–i5: 1 Drive folder each | The file/folder URL in each `instance_N/task.md` |
| `sheets_7_running_analysis` | i1–i5 | i1/i2: 1 Drive folder each; i3–i5: 1 source spreadsheet each | The URL in each `instance_N/task.md` |
| `sheets_10_paper_sorting` | i1–i5 | 2 Drive folders per instance (papers inbox + destination) | Both folder URLs in each `instance_N/task.md`; also re-run `setup_run.py` after substituting (it provisions the run folders) |
| `slides_17_removeimagesaddplaceholders` | i1–i5 | 1 source presentation + 2 Drive folders per instance | The three URLs in each `instance_N/task.md`, **and** the `DRIVE_FOLDER_ID` constant near the top of each `instance_N/evaluator.py` (the evaluator reads that folder directly) |
| `slides_30_Work_Wikipedia_Photos` | i1–i5 | 1 client-list Google Doc per instance | The Doc URL in each `instance_N/task.md`, **and** pass its ID as the evaluator's `--client_doc_id` argument |

Reference inventory (original asset URLs, so you can map zip contents to instances):

<details>
<summary>Original asset URLs per instance</summary>

- **docs_1_formal_letter**: i1 `folders/1Fonh_b2Gj9bsDl_J8Yu9C7OFI7GhdIHo` · i2 `folders/18Tc3la87mgvAbOG-EmkFXLfgf5O8DpvJ` · i3 `folders/1JCFk0TdlCPk-__Od8w_5ZKypb_gFVOKS` · i4 `folders/1qFCwye5C9Y2CkmKdcFE4q-5u5AqvQQ2T` · i5 `folders/1JCx-ULf0uOLh-VlF4hiv6y-AjUZnqxnp`
- **docs_11_personal_recipe_ocr**: i1 `file/d/1_7J7EoBKnjb73SyESJOzIK_TThKrQOtA` · i2 `folders/1JOT3oiD9G12Mge-RNtNVsIO73ewbRKnq` · i3 `folders/1bD6pJj-u4uJNYpbYDK0Dx58CoNq43T-W` · i4 `folders/18LsMdpjrprtlBzGGzmNft40yShaequD8` · i5 `folders/1Z5rf0sOtuEHxe4WdwDiXGpZTSZd8waGf`
- **sheets_7_running_analysis**: i1 `folders/1FgSMQLB-BKLGkniImqJrru8FnHLqpJwK` · i2 `folders/1lILPHzXjptPC0RJmoyt8ZusVEWmtYV68` · i3 `spreadsheets/d/1f0avjmLDiusBEIaS8GhE5LHpYY-vUCIcwl9LEPsS4Cc` · i4 `spreadsheets/d/1FzcltyDyqd30B7N2fwWsHbEuwSEZlECOu6j_KhM2UrM` · i5 `spreadsheets/d/1X83hQNelJfgd2ykTtgTIU55dK7QpQOewJSYqS-wElbU`
- **sheets_10_paper_sorting**: i1 `folders/1ID4WRSo5Zs9tfFzf6NFVUF5PbuesyuZE` + `folders/1dfRMRjBHH4F1S9WMD6p6VqpYQZ-pbKWB` · i2 `folders/1Qm2gLrC3PhRqhlAI_WXBjYKqECdPOwBE` + `folders/1lwLNgxT9_S6SKW43Ag-lyed5qPrGLfcP` · i3 `folders/1Fc1GthzO8dAuekt-L3dL4FfUbW7wjeZM` + `folders/1_QVWKSPZyaPXAEwypKnz-N3p4lQTztE-` · i4 `folders/1NIx27u2aOywiZRzBNaeucR4x7yWNf8YB` + `folders/1dTayId1h5Ft9QFeujrQ1kzsAANtdr4Nm` · i5 `folders/1Hp0H6hKTBNTxeIU3qXN0giqeVbvhaEXm` + `folders/1xDDOPz_AH55ONpjQRgC__IWjgWtbw08O`
- **slides_17_removeimagesaddplaceholders**: i1 `presentation/d/1TXzqKPonpldduitl5DOx_nIzdP9MWDcVgJzS2K7-HG4` + `folders/1Y5AfcruuTjBuDDvSkx7Tcl9rSeKe6e57` + `presentation/d/1rbHYFYmJKR_sxJYYYdAReFyPvFR3U_cQMk3TlvZwg9Y` · i2 `presentation/d/1g60_v3rq7JNESy-uTCZCabiOaOPldQI8zfGYcFeGGvs` + `folders/14RY4ExI1ZIrh-b7BsfdOkplYO-3H5nzK` + `folders/1h5YHAUzZnzT-QIDJbbkSytWZTgzMP_dO` · i3 `presentation/d/1V6r5Hg17LBtBYW5DBwmY4hW6eK3MYtnId9JgWw0cDvA` + `folders/1ZWlBfRO48joOLT6_NwyFf5HtCqwrOWeh` + `folders/1d5kreE1IYMJMfEOJ9ak2LKrPQchdsZiS` · i4 `presentation/d/1E49YKdl9qM1UwIx0bcAV1TX3RXE7rK09kefy3WhbttE` + `folders/1nUmoS50RQQMGDbXok6jyntpLXCjb6tId` + `folders/1sZUeENx2F8tVnDuHgJP6N9yp7fCD7qPo` · i5 `presentation/d/1TPE3rEGz7o8ZVW0NpFXrq4sZeeG_s-he4225lIMYxw0` + `folders/1MN-z56Qp8k3ThsREABxABrT4uHJfhnlB` + `folders/1PUOBpjNGdmrbEIA9c8TaeBzoKyqB5x0n`
- **slides_30_Work_Wikipedia_Photos**: i1 `document/d/1nGgs_hLHnORVZUmjbExF9BQzow5l-2VkE4jbdMdaPHM` · i2 `document/d/1UbVGCW8E-aEN367nuFPo3QUdlIAXQ0SjAA70UsG9Ez8` · i3 `document/d/1HMSnym7_9pbE1nuE05K5QTmWw8dKStCOux1CKGdb45Q` · i4 `document/d/1HG9ElXyInjFm_kpy1ZUQn73VG6FFuQ1IePdQqnAjZAs` · i5 `document/d/1DNTuS6T5KCVLIK4e4ysuvLyf8zt-4ni4xRPg92WNF-k`

</details>

All other task families are self-contained: agents research the live public web, and evaluators use only the bundled `data/` gold assets plus public APIs.
