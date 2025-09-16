## The official implemention of "FedSOV: A Secure Ownership Verification Scheme for Large-Scale Federated Learning System"

> **Abstract**: Federated learning (FL) enables collaborative training of a shared global model while preserving data confidentiality among participating entities. The substantial computational resources required for training and the considerable commercial value of the resulting model have driven the demand for reliable ownership verification mechanisms in federated learning systems. Current verification approaches predominantly utilize model watermarking techniques, where ownership claims are validated by detecting predefined watermarks embedded during the training process.
    However, these schemes are difficult to support large-scale federated learning systems with numerous clients due to the contradiction between the multi-client watermarks. Meanwhile, they overlook a common yet impactful attack known as the ambiguity attack, in which the attacker is able to forge a valid watermark and claim his ownership.
    To address these limitations, we propose a cryptographic signature-based federated learning secure model ownership verification scheme named FedSOV. FedSOV allows large numbers of clients to embed their ownership credentials and verify ownership using unforgeable digital signatures. The scheme provides theoretical resistance to ambiguity attacks with the unforgeability of the signature. We evaluate the fidelity, reliability, robustness, security, and time cost of FedSOV through experimental results on computer vision and natural language processing tasks. These results demonstrate that FedSOV is an effective ownership verification scheme for federated models, enhanced with provable cryptographic security.

<img width="988" height="413" alt="An illustration of FedSOV" src="https://github.com/user-attachments/assets/4c5f0b68-729a-4e30-b167-5a63169cd1a3" style="zoom: 50%;" />

### Prepration

Before executing the project code, please prepare the Python environment according to the `requirement.txt` file. We set up the environment with `python 3.8` and `torch 1.8.1`. 

### How to Run

The basic run command has been placed in the `run.sh` file. Use `bash run.sh` to run it. The model category, training set, watermark length, number of clients, GPU device, etc. can be set by yourself. For more parameter settings, please refer to the `utils/args.py` file.

