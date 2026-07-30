# Q5: Plain-English Guide (Explain Like I'm 10!) 🎓

A simple, intuitive guide to understanding **Question 5**: What we are building, why we use specific datasets, and how our smart clinical decision engine works.

## 1. The Big Picture: The Medical Puzzle 🧩

Imagine a hospital waiting room filled with **100 new skin cancer (melanoma) patients**. 

Doctors currently have **three main weapons (treatment options)** to fight melanoma:

1. **Arm A — Immunotherapy**: Giving drugs (`CD274` / anti-PD-1) that wake up the body's own immune system "soldier" cells (T-cells) so they hunt down and destroy cancer cells.
2. **Arm B — Targeted Therapy**: Using precision "sniper" drugs (`BRAF`/MEK inhibitors) that switch off a specific broken gene (`BRAF` mutation) inside the tumour.
3. **Arm C — Chemotherapy & Combination Therapy**: Using chemotherapy (dacarbazine) or adding a helper drug to "reset" the tumour environment.

### The Problem ⚠️
If doctors give **Immunotherapy to every single patient**, it only works for about **30 to 40 out of 100 people**. For the other 60 people, immunotherapy fails, precious weeks are lost while the tumour grows, and patients suffer unnecessary side effects.

### Our Goal in Q5 🎯
We are building a smart **"Sorting Hat" (Clinical Decision Engine)**. When a new patient arrives at the clinic, our software looks at their tumour's DNA, RNA expression, and immune cells, and tells the oncologist:
> *"This specific patient has an 80% chance of responding to Immunotherapy (Arm A), while that patient should go straight to Targeted Therapy (Arm B)!"*

## 2. Why We Use Two Different Datasets: The "Training School" vs. "Real World" 🏫

You might wonder: *Why do we have two datasets (`merged/immunotherapy` with 326 patients, and `merged/full` with 699 patients)?*

Think of it like **training a sniffer dog**:

```
┌──────────────────────────────────────────────────┐
│     DATASET 1: merged/immunotherapy (N=326)      │
│             "THE TRAINING SCHOOL"                │
├──────────────────────────────────────────────────┤
│ • Every patient received Immunotherapy           │
│ • We KNOW if treatment succeeded or failed       │
│ • Used to TRAIN AI to spot response patterns     │
└──────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│          DATASET 2: merged/full (N=699)          │
│           "THE REAL-WORLD HOSPITAL"              │
├──────────────────────────────────────────────────┤
│ • ALL melanoma patients (unselected population)  │
│ • Includes surgery, chemo, and targeted arms     │
│ • Used to TEST "Sorting Hat" treatment selection │
└──────────────────────────────────────────────────┘
```

1. **Dataset 1: `merged/immunotherapy` (326 Patients = The Training School)**
   * **Why**: To teach our AI what an *"Immunotherapy-Responsive Tumour"* looks like vs. an *"Immune-Resistant Tumour"*, we need patients who **actually received immunotherapy** and where we know the true outcome (`Responder` vs `Non-Responder`). 
   * This is where the AI learns that high T-cells and high M1 macrophages mean high response probability!

2. **Dataset 2: `merged/full` (699 Patients = The Real-World Hospital)**
   * **Why**: Once our AI finishes "Training School", we test its treatment selection logic on a real-world, unselected group of 699 melanoma patients (including those with untreated primary tumours or on other therapies). 
   * The decision engine sorts all 699 patients into Arm A (Immunotherapy), Arm B (Targeted Therapy), or Arm C (Chemotherapy / Combination).

> [!NOTE]
> **Data Harmonisation & DNA Chromosome Note**: We use iAtlas harmonised data to keep gene expression measurements identical across all datasets without technical noise. Because raw trial datasets lack Copy Number Alteration (CNA) chromosome maps, we track DNA instability via Tumour Mutational Burden (TMB) and inspect gene loss (`PTEN`, `CDKN2A`) via gene mutation + mRNA expression levels.

## 3. The Macrophage Scorecard: Good Cops vs. Bad Cops 🚓

Inside a tumour, there are immune cells called **macrophages**. But they act like two completely different police officers:

* 👮 **M1 Macrophages ("Good Cops")**: Pro-inflammatory cells that actively attack and kill tumour cells.
* 🦹 **M2 Macrophages ("Double Agents")**: Anti-inflammatory cells that protect the tumour from T-cells and help it grow.

### What is the Macrophage STV?
The project uses a special 14,837-gene scorecard ([m1_m2_stv.csv](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/data/config/m1_m2_stv.csv)). We use this matrix to calculate an **M1/M2 Ratio** (`M1 / (M1 + M2)`) for every patient:

* **High M1 / Low M2**: Tumour is filled with "Good Cops" $\rightarrow$ **High response to Immunotherapy!**
* **Low M1 / High M2**: Tumour is guarded by "Double Agents" $\rightarrow$ Immunotherapy alone will fail, but **adding an M2-blocking helper drug** can convert the patient into a responder!

## 4. How the "Sorting Hat" Decision Flow Works 🧙‍♂️

When any random new patient enters the clinic, Q5 runs them through a simple 3-step decision tree:

```
             ┌────────────────────────────────────┐
             │         RANDOM NEW PATIENT         │
             │   (Tumour Gene Expression & DNA)   │
             └─────────────────┬──────────────────┘
                               │
                               ▼
        [Step 1] Is Immunotherapy Chance High (>70%)?
          ├── YES ──► 🟢 ARM A: Immunotherapy (Anti-PD-1)
          └── NO  ──► Proceed to Step 2
                               │
                               ▼
        [Step 2] Is the `BRAF` Gene Broken (`BRAF` V600)?
          ├── YES ──► 🔵 ARM B: Targeted Therapy
          └── NO  ──► Proceed to Step 3
                               │
                               ▼
        [Step 3] Can We Fix Immune Barrier (Treatability)?
          ├── YES ──► 🟡 COMBINATION (Helper Drug + IO)
          └── NO  ──► 🔴 ARM C: Chemotherapy / Trial
```

## 5. How All 5 Assignment Questions Fit Together 🧩

Each of the 5 project questions answers one piece of the puzzle:

* 📈 **Q1 (The Response Predictor)**: Calculates the percentage chance ($0-100\%$) that immunotherapy will work for a patient.
* 🧪 **Q2 (The Cell Line Lab)**: Predicts how sensitive cancer cells are to targeted & chemotherapy drugs.
* ⏱️ **Q3 (The Time-Travel Simulator - ODEs)**: Simulates a graph showing how tumour size ($T(t)$) shrinks or grows over 180 days under different treatments.
* 🎯 **Q4 (The Drug Hunter - DepMap & LINCS)**: Mines databases to find new secret-weapon drugs to wake up "cold" tumours.
* 🧠 **Q5 (The Master Brain)**: Combines Q1, Q2, Q3, and Q4 into one clean, easy-to-use clinical recommendation tool for doctors!

## Summary Checklist 

| Concept | Plain-English Meaning | Why It Matters |
| :--- | :--- | :--- |
| **`merged/immunotherapy` ($N=326$)** | The "Training School" dataset | Teaches AI what immunotherapy response looks like |
| **`merged/full` ($N=699$)** | The "Real Hospital" dataset | Tests how AI sorts an unselected population |
| **Macrophage STV** | Good Cop vs. Bad Cop score | Identifies patients needing M2-blocking drugs |
| **Q3 ODEs** | Tumour growth time-machine | Shows predicted tumour shrinkage over 180 days |
| **Q4 DepMap/LINCS** | Novel drug target finder | Recommends helper drugs for resistant patients |
| **Q5 Engine** | Master Sorting Hat | Assigns patients to Arm A, Arm B, or Arm C |
