# LoreForge: Multi-Universe Lore-Faithful Story Generation

## Datasets

### Pretraining

| Dataset | Source | License | Use |
|---|---|---|---|
| `manu/project_gutenberg` | HuggingFace | Public domain | Base model pretraining on general English prose (~70k novels) |

---

### Star Wars

| Dataset | Source | License | Use |
|---|---|---|---|
| `lara-martin/Scifi_TV_Shows` | HuggingFace | CC-BY-4.0 | Fine-tuning + RAG — ~270 Star Wars stories scraped from the Star Wars Fandom wiki, filtered by keyword. Provides lore prose sentences covering characters, events, and locations. |

---

### Harry Potter

| Dataset | Source | License | Use |
|---|---|---|---|
| `rupanshukapoor/harry-potter-books` | Kaggle | MIT (educational/research only) | Fine-tuning + RAG — Full text of all seven HP books as plain .txt files (~2.5 MB). Teaches narrative style and provides retrievable passages covering characters, spells, locations, and events. |

---

### Lord of the Rings

| Dataset | Source | License | Use |
|---|---|---|---|
| `jeremyarancio/lotr-book` | HuggingFace | Unstated (educational/research only) | Fine-tuning — Full LOTR trilogy text (pages 45–1055). Teaches the LoRA adapter Tolkien's prose style: archaic diction, elevated register, and the narrative rhythm of Middle-earth. |
| `wikimedia/wikipedia` (filtered) | HuggingFace | CC BY-SA 3.0 | RAG — English Wikipedia filtered to LOTR-related articles (characters, locations, factions, artifacts). Encyclopedic structure makes these ideal retrieval chunks for grounding generation in canon facts. |
