# Follow-up evaluation: memory run 2

policy_rag 3.3.0 | 2026-10-27T21:24:28 | dialogues `optima_secure_followups.json` version `fbf48a727e` | rewrite template `condense_v2` | test set `7313dd7571`

**Decision (rule fixed before the run):** DO NOT ADOPT MEMORY: follow-ups not significantly better with memory (fixed 6, broken 1, p = 0.125); 1 harm-check turn(s) (self-contained follow-up or topic switch) became wrong: Q28.2

Test follow-ups that need the conversation: 28; fully correct without memory 16, with memory 21, as the complete question 28. Fixed by memory: Q04.2, Q04.3, Q06.2, Q22.2, Q23.2, Q24.2; broken: Q13.2; McNemar exact p = 0.1250. Harm-check turns (self_contained, topic_switch) made wrong: Q28.2 of 14.

## Fully correct by condition (test split)

| condition | follow_up | after_refusal | self_contained | topic_switch | deciding (follow_up + after_refusal) |
|---|---|---|---|---|---|
| no_memory | 16/27 | 0/1 | 8/8 | 6/6 | 16/28 (39–73%) |
| memory | 20/27 | 1/1 | 7/8 | 6/6 | 21/28 (57–87%) |
| gold | 27/27 | 1/1 | 8/8 | 6/6 | 28/28 (88–100%) |

Rewrites: 32 of 42 turns changed; 0 not sent to the rewriter; 2 rejected by the checks. Rewrite time (seconds, generated): median 38.1, max 862. Total run time 129 min.

## Turns

| turn | kind | asked | searched_as | fallback | correct_no_memory | correct_memory | correct_gold | memory_issues |
|---|---|---|---|---|---|---|---|---|
| Q01.2 | follow_up | Does it still apply if I have been continuously insured for more than a year? | Does the waiting period still apply if I have been continuously insured for more than a year? |  | True | True | True |  |
| Q02.2 | follow_up | And for other illnesses in the first days after the policy starts? | And for other illnesses in the first days after the policy starts? |  | True | True | True |  |
| Q03.2 | follow_up | Is there a deductible on those claims? | Is there a deductible on those claims? | rewrite copies an earlier answer: 'the global health cover emergency treatments ...' | True | True | True |  |
| Q04.2 | follow_up | What does the first one exclude? | What does the Global Health Cover for emergency treatments only exclude? |  | False | True | True |  |
| Q04.3 | follow_up | And which benefits does the second one extend outside India? | And which benefits does the Global Health Cover for emergency and planned treatments extend outside India? |  | False | True | True |  |
| Q05.2 | follow_up | What happens to that extra if I reduce my Sum Insured at renewal? | What happens to the Plus Benefit if I reduce my Sum Insured at renewal? |  | True | True | True |  |
| Q06.2 | follow_up | Does any unused part of it carry forward to the next year? | Does any unused part of the Secure Benefit carry forward to the next year? |  | False | True | True |  |
| Q07.2 | follow_up | Which covers is it not applied to at all? | Which covers is the Aggregate Deductible not applied to at all? |  | True | True | True |  |
| Q08.2 | follow_up | Which illnesses does it treat as major? | Which illnesses does the cover treat as major? |  | True | True | True |  |
| Q09.2 | follow_up | How long do I have to be admitted for a claim to count? | How long do I have to be admitted for the benefit to count? |  | False | False | True | not in the context: 24 consecutive hours / not in the context: day care / refused by the relevance gate |
| Q09.3 | follow_up | Does it also pay for the ambulance that takes me there? | Does the plan also pay for ambulance expenses? |  | True | True | True |  |
| Q10.2 | follow_up | What does it not pay for? | What does the policy not pay for? |  | False | False | True | not in the context: pre hospitalization / search did not find: transportation or preservation / search did not find: organ harvesting |
| Q11.2 | follow_up | And for a planned admission, how far ahead? | And for planned hospitalization, how far ahead? |  | True | True | True |  |
| Q12.2 | follow_up | Once I have sent all the papers, how soon will I get the money? | Once I have sent all the documents, how soon will I get the money? |  | True | True | True |  |
| Q13.2 | follow_up | Once the approval is given, how long is it valid? | Once the approval is given, how long is the approval valid? |  | True | False | True | model refused |
| Q14.2 | follow_up | What is the most I can save if I combine it with other ones? | What is the most I can save if I combine the online discount with other applicable discounts? |  | False | False | True | figures not in the context: 72.5% |
| Q14.3 | follow_up | Is there a special one for NRI policyholders? | Is there a special discount for NRI policyholders? |  | True | True | True |  |
| Q15.2 | follow_up | If I do that, do I keep the waiting period I have already served? | If I do the portability request, do I keep the waiting period I have already served? |  | False | False | True | in the context, not in the answer: continuity |
| Q16.2 | follow_up | What happens to my son's cover when he reaches that age? | What happens to my son's cover when he reaches that age? | rewrite adds figures that are not in the user's questions: 25 | True | True | True |  |
| Q17.2 | follow_up | Do those rates include GST? | Does the premium include GST? |  | True | True | True |  |
| Q17.3 | follow_up | Under which section of the Income Tax Act can I claim it as a deduction? | Under which section of the Income Tax Act can I claim the premium as a deduction? |  | True | True | True | cites section numbers not in the context: 80 |
| Q18.2 | follow_up | Is it paid for days spent in the ICU? | Is the cash benefit paid for days spent in the ICU? |  | True | True | True |  |
| Q19.2 | follow_up | What exactly does that cover include? | What exactly does the policy cover include? |  | False | False | True | in the context, not in the answer: sub limit / says something is not shown, but the context has: sub limit |
| Q20.2 | follow_up | Which ones are offered? | Which Add-on Covers are offered? |  | True | True | True |  |
| Q21.2 | follow_up | And if I pay first and claim later, which extra papers do I need? | And if the insured person pays first and claims later, which extra documents are required? |  | False | False | True | in the context, not in the answer: passport / in the context, not in the answer: flight tickets / in the context, not in the answer: accommodation invoices / in the context, not in the answer: written advice |
| Q22.2 | follow_up | How is it different from the version that also covers planned treatments? | How is the Global Health Cover (Emergency Treatments Only) different from the plan that also covers planned treatments? |  | False | True | True |  |
| Q23.2 | follow_up | How does that work when the whole family shares one floater sum insured? | How does the Cumulative Bonus work when the whole family shares one floater sum insured? |  | False | True | True |  |
| Q24.2 | after_refusal | Then what would Optima Secure with Rs. 10 lakh cover cost at that age in Mumbai? | What is the annual premium for Optima Secure with Rs. 10 lakh cover for a 30-year-old in Mumbai? |  | False | True | True |  |
| Q25.2 | self_contained | Can they refuse to renew my policy because I made claims? | Can the insurer refuse to renew the policy because claims were made? |  | True | True | True |  |
| Q26.2 | self_contained | If I cancel the policy later, how much notice do I give and what is refunded? | If I cancel the plan later, how much notice do I give and what is refunded? |  | True | True | True |  |
| Q27.2 | self_contained | How many adults and children can a family floater policy cover? | How many adults and children can a family floater policy cover? |  | True | True | True |  |
| Q28.2 | self_contained | Can my 3-year-old be insured without a parent on the policy? | Can the dependent child be insured without a parent being included in the policy? |  | True | False | True | in the context, not in the answer: 5 years / answers 'yes', the policy says 'no' |
| Q29.2 | self_contained | So can I claim for new spectacles? | Can I claim for new spectacles? |  | True | True | True |  |
| Q30.2 | self_contained | What is the maximum medical underwriting loading they can apply? | What is the maximum medical underwriting loading they can apply? |  | True | True | True |  |
| Q31.2 | self_contained | So which hospitalization expenses does the policy pay? | So which hospitalization expenses does the plan cover? |  | True | True | True |  |
| Q32.2 | self_contained | Am I covered if I get hurt scuba diving as a professional diver? | Am I covered if I get hurt scuba diving as a professional diver? |  | True | True | True |  |
| Q33.2 | topic_switch | Which cities are classified as Tier 1 for premium calculation? | Which cities are classified as Tier 1 for premium calculation? |  | True | True | True |  |
| Q34.2 | topic_switch | What is the moratorium period? | What is the moratorium period? |  | True | True | True |  |
| Q35.2 | topic_switch | What is the Preventive Health Check-up benefit? | What is the Preventive Health Check-up benefit? |  | True | True | True |  |
| Q36.2 | topic_switch | Within how many days must the company settle a claim after receiving the last necessary document? | Within how many days must the insurer settle a claim after receiving the last necessary document? |  | True | True | True |  |
| Q37.2 | topic_switch | What is domiciliary hospitalization and when is it covered? | What is domiciliary hospitalization and when is it covered? |  | True | True | True |  |
| Q38.2 | topic_switch | Does this policy cover treatment taken outside India? | Does the policy cover treatment taken outside India? |  | True | True | True |  |
