# Dataset Integrity & Leakage Resolution Audit Summary

## Executive Overview
A rigorous, multi-stage data integrity audit was conducted to identify and eliminate potential perceptual near-duplicates and cross-split data leakage across training, validation, and test partitions.

---

## 1. Audit Methodology & Perceptual Hashing
- **Algorithm**: Difference Hashing (dHash) computed on normalized 64-bit gradient representations.
- **Hamming Distance Threshold**: Pairs with a Hamming distance $\le 3$ were flagged as potential near-duplicates.
- **Candidate Evaluation**: Approximately 7,422 cross-split candidate pairs were evaluated across splits.
- **Human Review**: Category-based human forensic review of high-probability candidate pairs was performed to distinguish true duplicate smears/crops from distinct cells of identical morphology.

---

## 2. Leakage Remediation & Split Isolation
1. **Cluster Detection**: Graph-connected components were formed for all confirmed and likely duplicates.
2. **Partition Rebalancing**: Any duplicate clusters spanning split boundaries (e.g. train $\leftrightarrow$ test or train $\leftrightarrow$ val) were resolved by consolidating all associated samples into a single split or pruning leaked instances.
3. **Cryptographic Verification**: Post-remediation splits were cryptographically verified using SHA-256 manifest hashing.
4. **Final Certified Partitions**:
   - Training Set: Cleaned, leak-free training partition
   - Validation Set: Independent validation partition (macro F1: 96.84%)
   - Test Set: 5,000 strictly isolated single-cell images with zero cross-split leakage (accuracy: 97.02%, macro F1: 97.04%)

---

## 3. Privacy & Sanitization Notice
> **Data Protection Compliance**:
> In accordance with medical data privacy guidelines and research integrity protocols, raw medical images, individual patient identifiers, local directory hierarchies, and raw duplicate-resolution CSV files containing image filenames are strictly excluded from this repository.
