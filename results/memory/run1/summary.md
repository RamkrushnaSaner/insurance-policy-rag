# Follow-up evaluation: memory run 1

policy_rag 3.3.0 | 2026-10-27T11:08:23 | dialogues `optima_secure_followups.json` version `861c4640a2` | rewrite template `condense_v1` | test set `7313dd7571`

**Decision (rule fixed before the run):** DO NOT ADOPT MEMORY: follow-ups not significantly better with memory (fixed 2, broken 1, p = 1.000)

Test follow-ups that need the conversation: 16; fully correct without memory 11, with memory 12, as the complete question 15. Fixed by memory: D01.2, D11.2; broken: D23.2; McNemar exact p = 1.0000. Harm-check turns (self_contained, topic_switch) made wrong: none of 6.

## Fully correct by condition (test split)

| condition | follow_up | after_refusal | self_contained | topic_switch | deciding (follow_up + after_refusal) |
|---|---|---|---|---|---|
| no_memory | 10/14 | 1/2 |  | 4/6 | 11/16 (44–86%) |
| memory | 12/14 | 0/2 |  | 4/6 | 12/16 (51–90%) |
| gold | 13/14 | 2/2 |  | 4/6 | 15/16 (72–99%) |

Rewrites: 18 of 26 turns changed; 0 not sent to the rewriter; 0 rejected by the checks. Rewrite time (seconds, generated): median 47.6, max 382. Total run time 68 min.

## Turns

| turn | kind | asked | searched_as | fallback | correct_no_memory | correct_memory | correct_gold | memory_issues |
|---|---|---|---|---|---|---|---|---|
| D01.2 | follow_up | And how many days after discharge? | How many days after discharge are covered? |  | False | True | True |  |
| D02.2 | follow_up | And for post-hospitalization? | And for post-hospitalization expenses? (up to 60 days after the date of discharge) |  | False | True | True |  |
| D03.2 | follow_up | How does that differ for monthly and yearly instalments? | How does that differ for monthly and yearly instalments? |  | True | True | True |  |
| D04.2 | follow_up | What can the company do if I didn't disclose one? | What can the company do if I didn't disclose a pre-existing disease? |  | True | True | True |  |
| D05.2 | follow_up | Does it pay for gloves and food charges in a hospitalization claim? | Does the Protect Benefit pay for gloves and food charges in a hospitalization claim? |  | True | True | False |  |
| D06.2 | follow_up | And for a single claim on Optima Secure, how does it change from Year 1 to Year 3? | And for a single claim on Optima Secure, how does it change from Year 1 to Year 3? |  | False | False | False | in the context, not in the answer: 25 lakh / table values missing: 1 of 3 |
| D07.2 | follow_up | What happens to it if the Sum Insured is reduced at renewal? | What happens to the Plus Benefit if the Sum Insured is reduced at renewal? |  | True | True | True |  |
| D08.2 | follow_up | On a family floater, how is it added and when is it reduced? | On a family floater, how is the Cumulative Bonus added and when is it reduced for the Optima Suraksha plan? |  | True | True | True |  |
| D08.3 | follow_up | And what happens to it if a claim from the expiring year is notified after the renewal premium is accepted? | And what happens to the Cumulative Bonus if a claim from the expiring year is notified after the renewal premium is accepted for the Optima Suraksha plan on a family floater? |  | True | True | True |  |
| D09.2 | follow_up | What is the per claim deductible under it? | What is the per claim deductible under the Global Health Cover (Emergency Treatments Only)? |  | True | True | True |  |
| D09.3 | follow_up | What limits and conditions apply to its claims? | What limits and conditions apply to the claims under the Global Health Cover (Emergency Treatments Only)? |  | False | False | True | in the context, not in the answer: base sum insured and plus benefit / says something is not shown, but the context has: base sum insured and plus benefit |
| D10.2 | follow_up | What is the procedure for cashless claims there? | What is the procedure for cashless claims under the Global Health Cover (Emergency & Planned Treatments) outside India? |  | False | False | True | in the context, not in the answer: network provider / says something is not shown, but the context has: network provider |
| D10.3 | follow_up | And which additional documents are needed for such a claim? | Which additional documents are needed for cashless claims under the Global Health Cover (Emergency & Planned Treatments) outside India? |  | True | True | True |  |
| D11.2 | follow_up | How does cashless approval work for it? | How does cashless approval work for Home Health Care cover? |  | False | True | True |  |
| D12.2 | follow_up | And what is the maximum entry age for a dependent child? | And what is the maximum entry age for a dependent child, which is 25 years? |  | True | True | True |  |
| D13.2 | follow_up | What is the sub-limit for it per eye? | What is the sub-limit for cataract surgery per eye? |  | True | True | True |  |
| D14.2 | follow_up | And for cashless claims in India? | And for cashless claims in health insurance in India? |  | True | True | True |  |
| D15.2 | topic_switch | Does unused Secure Benefit carry forward to the next year? | Does unused Secure Benefit carry forward to the next year? |  | True | True | True |  |
| D16.2 | topic_switch | Is dental treatment covered? | Is dental treatment covered? |  | False | False | False | not in the context: disease or injury |
| D17.2 | topic_switch | Is cataract surgery covered from day one? | Is cataract surgery covered from day one? |  | False | False | False | not in the context: 24 months / model refused |
| D18.2 | topic_switch | How long is the free look period? | How long is the free look period? |  | True | True | True |  |
| D19.2 | topic_switch | Can I pay my premium every month? | Can I pay my premium every month? |  | True | True | True |  |
| D20.2 | topic_switch | What is the moratorium period? | What is the moratorium period? |  | True | True | True |  |
| D21.2 | after_refusal | So will the policy pay for my wife's delivery? | Will the policy pay for my wife's delivery after the 9-month waiting period for maternity benefits? |  | False | True | False |  |
| D22.2 | after_refusal | What about Optima Secure with a Rs. 10 lakh Sum Insured for a 30-year-old living in Mumbai? | What about Optima Secure with a Rs. 10 lakh Sum Insured, Tier 1, for a 30-year-old living in Mumbai? |  | False | False | True | search did not find: 13,500 |
| D23.2 | after_refusal | Then what is the Automatic Restore Benefit? | What is the Automatic Restore Benefit under the Optima Secure policy, including its coverage of 100% of the Base Sum Insured for partial or complete utilization within India under Section 0? |  | True | False | True | figures not in the context: 100% |
