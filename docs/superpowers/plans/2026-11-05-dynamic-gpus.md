# Dynamic GPU queue implementation plan

The user explicitly authorizes periodically detecting free GPUs and parallel training on available GPUs 0/1/2/3. This supersedes GPU3-only execution, preserving the frozen model/data/search protocol.

Use one CPU supervisor (30-second polling), one isolated process per assigned GPU, per-job locks, and an exclusive publication lock. Require two consecutive observations with utilization <=10%, memory <=1024 MiB, >=12 GiB free, and no non-desktop compute applications before using an additional GPU. Never terminate other users' jobs. GPU3 is the already-authorized base slot, subject to the free-memory floor.

Keep the original GPU3 process running while no additional card is free. When a spare qualifies, pause the legacy parent only at a completed-job boundary, allow its publication subprocess to finish, retire that parent, and acquire its global study lock. Dispatch remaining jobs from the unchanged frozen queue. Do not alter frozen training/model files or historical protocol signatures. Each new worker records actual physical GPU/UUID, train/test execution and supervisor/entrypoint digests separately.

- [ ] Test idle detection, desktop exceptions, failures, duplicate job locks, queue filtering, and boundary-only handover using CPU fixtures.
- [ ] Implement supervisor and single-job worker entrypoint using existing train_job/test_job unchanged.
- [ ] Serialize publication against completion-marker writes so independent audits see a consistent snapshot.
- [ ] Surface execution policy and active GPUs in paper evidence/status; document runtime controls.
- [ ] Run full tests and a read-only review; start a detached monitor and verify repeated polling without interrupting the current training job.

Failure policy: recording/code/data incompatibility stops new dispatch; failed jobs and publication failures are explicit and never counted complete. Already-running jobs may finish. Idle detection errors cannot launch work. The supervisor must be restartable and must not duplicate surviving worker processes.
