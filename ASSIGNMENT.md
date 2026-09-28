# Original Assignment Brief

Preserved verbatim from the initial README.md, before it was replaced with
project-specific documentation (see [README.md](README.md)).

---

# Week-03-Adapter-Sprint-LoRA-QLoRA-Instruction-Tuning-and-DPO-Under-a-15-GB-Memory-Ceiling
Week 03 Assessment
🏙️ CivicDesk 311 Complaint Rewriter: NLP Project Brief
👤 Role & Context
Company: CivicDesk (A vendor routing municipal 311 service requests for mid-sized cities)
Role: Junior NLP Engineer
Stakeholder: Elena Vasquez (Product Manager)
🎯 Objective
Build a lightweight AI assistant that automatically rewrites messy, unstructured resident complaints into clean, structured dispatch tickets.
⚠️ Constraints
Hardware Budget: Strictly limited to a free-tier Google Colab GPU.
Tooling: Must utilize Claude Code or Claude Co-Work for development and agent orchestration.
🛠️ Execution Steps
1. Model Selection & Fine-Tuning
Select a small, open-source pretrained model (e.g., a sub-1B parameter model from the Hugging Face Hub).
Adapt the model using LoRA or QLoRA to ensure it fits within the memory constraints of the free-tier GPU.
2. Instruction Dataset Generation
Create an instruction-tuning dataset consisting of a few hundred realistic 311 complaint-to-ticket pairs.
Action: Write a Python script or prompt the AI agent to synthesize this data.
3. Preference Optimization (DPO)
Construct a preference dataset containing chosen (ideal) vs. rejected (poor) rewrites.
Execute a short DPO (Direct Preference Optimization) pass on top of the LoRA adapter to align the model's outputs with CivicDesk's quality standards.
4. Instrumentation & Profiling
Rigorously track and log peak GPU memory usage for every training and inference run.
Track and log wall-clock time for every run to prove efficiency.
📦 Deliverables
Code Repository: A complete, well-documented repo containing the data synthesis scripts, LoRA/QLoRA training loops, DPO implementation, and memory/time instrumentation.
Narrated Session Recording: A video or audio walkthrough demonstrating the workflow, proving that the entire approach successfully fits within the free-tier Colab hardware budget.
Target Audience: Elena Vasquez and the CivicDesk platform team.
