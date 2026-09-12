# Legal Analysis: License Choice & Dependency Compatibility

**Last updated:** September 2026  
**License:** `LICENSE` (Finetune Studio Non-Commercial and Research License v1.0)

This document explains why this license was chosen, how it interacts with our
dependencies, and what it means for you. **It is not legal advice.** Consult a
lawyer for your specific situation.

---

## 1. The license in plain English

| Activity | Allowed? |
|---|---|
| Personal use, hobby projects | ✅ Yes |
| Academic & scientific research | ✅ Yes |
| Education & teaching | ✅ Yes |
| Publishing research using models trained with the Software | ✅ Yes |
| Non-commercial open-source projects | ✅ Yes |
| **Commercial use of the Software** | ❌ No — requires a paid commercial license |
| **Training models for commercial deployment** | ❌ No — requires a paid commercial license |
| SaaS / hosted inference as a business | ❌ No — requires a paid commercial license |
| Using the Software inside a for-profit company | ❌ No — requires a paid commercial license |

**In short:** Free for personal and science. Paid for commercial.

---

## 2. Why not an OSI-approved open-source license?

The Open Source Initiative's [Open Source Definition](https://opensource.org/osd)
(§6) prohibits licenses that discriminate against "fields of endeavor,"
including commercial use. The Free Software Foundation's [Four
Freedoms](https://www.org.org/philosophy/free-sw.html) likewise forbid
restrictions on commercial use or on running the program for any purpose.

A license that says "non-commercial only" or "no commercial model training"
**cannot** be OSI-approved or FSF-approved as free software.

We chose this deliberately. The goal is:

- Give individuals and researchers full freedom
- Prevent commercial entities from using the Software to build competing
  commercial model-training products without contributing back
- Reserve the right to charge for commercial use (which funds development)

**This is "source-available" software**, in the same family as:
- [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)
- [Meta Llama Community License](https://www.llama.com/llama3_2/license/) (non-commercial clause for large-scale commercial users)
- [BigCode OpenRAIL-M](https://huggingface.co/spaces/bigcode/openrail-license) (use restrictions)

---

## 3. Compatibility with our dependencies

Our license only covers *our* code. We do not relicense third-party packages.
Our dependencies remain under their original permissive licenses. This table
shows the compatibility analysis:

| Dependency | Their license | Restrictions? | Compatible with our license? |
|---|---|---|---|
| torch | BSD-3-Clause | Attribution | ✅ Yes |
| transformers | Apache-2.0 | Attribution, patent grant | ✅ Yes |
| peft | Apache-2.0 | Attribution, patent grant | ✅ Yes |
| trl | Apache-2.0 | Attribution, patent grant | ✅ Yes |
| accelerate | Apache-2.0 | Attribution, patent grant | ✅ Yes |
| datasets | Apache-2.0 | Attribution, patent grant | ✅ Yes |
| safetensors | Apache-2.0 | Attribution, patent grant | ✅ Yes |
| huggingface-hub | Apache-2.0 | Attribution, patent grant | ✅ Yes |
| sentence-transformers | Apache-2.0 | Attribution, patent grant | ✅ Yes |
| chromadb | Apache-2.0 | Attribution, patent grant | ✅ Yes |
| gguf | MIT | Attribution | ✅ Yes |
| llama-cpp-python | MIT | Attribution | ✅ Yes |
| bitsandbytes | MIT | Attribution | ✅ Yes |
| fastapi | MIT | Attribution | ✅ Yes |
| uvicorn | BSD-3-Clause | Attribution | ✅ Yes |
| jinja2 | BSD-3-Clause | Attribution | ✅ Yes |
| python-multipart | Apache-2.0 | Attribution, patent grant | ✅ Yes |
| pydantic | MIT | Attribution | ✅ Yes |
| aiofiles | Apache-2.0 | Attribution, patent grant | ✅ Yes |
| rich | MIT | Attribution | ✅ Yes |
| aiosqlite | MIT | Attribution | ✅ Yes |
| tiktoken | MIT | Attribution | ✅ Yes |
| requests | Apache-2.0 | Attribution, patent grant | ✅ Yes |
| numpy | BSD-3-Clause | Attribution | ✅ Yes |
| pandas | BSD-3-Clause | Attribution | ✅ Yes |
| scikit-learn | BSD-3-Clause | Attribution | ✅ Yes |
| scipy | BSD-3-Clause | Attribution | ✅ Yes |
| playwright | Apache-2.0 | Attribution, patent grant | ✅ Yes |
| JetBrains Mono, Share Tech Mono, VT323, Press Start 2P | OFL-1.1 | Attribution, no sale of font alone | ✅ Yes |

### Why this works

All our dependencies use **permissive** licenses (MIT, Apache-2.0, BSD-3-Clause,
OFL-1.1). Permissive licenses allow:

- Use, modification, and distribution of the dependency *itself*
- Use in proprietary/closed-source works
- Adding additional restrictions on *separate, combined* works (as long as the
  dependency's attribution is preserved)

None of our dependencies use **copyleft** licenses (GPL, AGPL, LGPL, MPL,
EPL). Copyleft would require derivative works to be distributed under the same
license — which would conflict with our non-commercial restriction.

**Result:** Our non-commercial restriction applies to *our code*, not to the
dependencies. Users can replace or redistribute the dependencies under their
original licenses. This is legally sound.

### What would NOT be compatible

If any of our dependencies were under GPL/AGPL, we could not apply a
non-commercial restriction to the combined work, because the GPL forbids
adding restrictions beyond what it permits. We would need to replace that
dependency or relicense our code under GPL (with the copyleft obligation).

We monitor this. If a new dependency introduces copyleft code, we will either
replace it or adjust our license accordingly.

---

## 4. Obligations for redistributors

If you redistribute the Software (modified or unmodified), you must:

1. **Include a copy of the `LICENSE`** file with the distribution
2. **Retain all copyright notices** in the source code
3. **Make the recipient aware** of the license terms (they must agree)
4. **Not impose additional restrictions** beyond what the license already
   imposes (no further DRM, no additional EULA that contradicts the license)

You may not sublicense the Software or grant rights you don't have.

---

## 5. Commercial licensing

To use the Software for commercial purposes, or to use it for commercial model
training, you need a separate written commercial license from the Licensor.

**Contact:**
- GitHub: https://github.com/GenorTG/finetune-studio
- Repository: `GenorTG/finetune-studio`

Commercial licenses are negotiated individually and may involve:
- A flat license fee
- Revenue sharing on models trained with the Software
- A subscription model
- Custom terms for specific use cases

We're happy to discuss. Reach out via GitHub issues or the repository contact.

---

## 6. Frequently asked questions

**Q: Can I use this for a university research project?**  
A: Yes. Academic research is explicitly a permitted non-commercial purpose.

**Q: Can I train a model and release it open-source for non-commercial use?**  
A: Yes. As long as the resulting weights are also non-commercial and you comply
with the license terms.

**Q: Can I use this inside my company to train models we sell?**  
A: No. That's commercial model training. You need a paid commercial license.

**Q: Can I use this to fine-tune a model for a non-profit organization?**  
A: Yes, if the use is genuinely non-commercial. If the non-profit generates
revenue from the model, contact us about a commercial license.

**Q: Is this legally enforceable?**  
A: We believe so, based on established copyright licensing principles and
precedents (e.g., the PolyForm licenses, the Meta Llama Community License,
the MIT/Apache families). However, no court has tested this specific license.
Consult a lawyer for certainty.

**Q: Why not just use MIT?**  
A: MIT would allow anyone (including large commercial entities) to use the
Software freely to build competing products, without contributing back. We
want to prevent that while keeping it free for individuals and researchers.

**Q: Will you ever relicense to a standard open-source license?**  
A: We may offer dual-licensing (open-source + commercial) in the future. For
now, the non-commercial restriction is intentional.

---

## 7. Disclaimer

This analysis is for informational purposes only. It is **not legal advice**.
The enforceability of license terms varies by jurisdiction. Consult a
qualified attorney for advice on your specific situation.

The Licensor makes no warranty about the accuracy or completeness of this
analysis.

---

*End of document.*
