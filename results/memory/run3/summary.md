# Follow-up evaluation: memory run 3

policy_rag 3.3.0 | 2026-10-29T04:40:07 | dialogues `optima_secure_followups.json` version `a9a5aa60b1` | rewrite template `condense_v3` | test set `7313dd7571`

**Decision (rule fixed before the run):** DO NOT ADOPT MEMORY: follow-ups not significantly better with memory (fixed 5, broken 3, p = 0.727)

Test follow-ups that need the conversation: 32; fully correct without memory 19, with memory 21, as the complete question 32. Fixed by memory: R08.2, R16.2, R26.2, R28.2, R31.2; broken: R01.2, R17.2, R21.2; McNemar exact p = 0.7266. Harm-check turns (self_contained, topic_switch) made wrong: none of 12.

## Fully correct by condition (test split)

| condition | follow_up | after_refusal | self_contained | topic_switch | deciding (follow_up + after_refusal) |
|---|---|---|---|---|---|
| no_memory | 18/30 | 1/2 | 5/6 | 6/6 | 19/32 (42–74%) |
| memory | 19/30 | 2/2 | 5/6 | 6/6 | 21/32 (48–80%) |
| gold | 30/30 | 2/2 | 6/6 | 6/6 | 32/32 (89–100%) |

Rewrites: 27 of 44 turns changed; 14 not sent to the rewriter; 3 rejected by the checks. Rewrite time (seconds, generated): median 43.8, max 330. Total run time 106 min.

## Turns

| turn | kind | asked | searched_as | fallback | correct_no_memory | correct_memory | correct_gold | memory_issues |
|---|---|---|---|---|---|---|---|---|
| R01.2 | follow_up | What about after I'm discharged? | What about the coverage after discharge? |  | True | False | True | uses another cover's or procedure's section: cites Section 4.10; cites Section 4.9 |
| R02.2 | follow_up | And how many days before admission? | How long is the waiting period for pre-admission expenses under the Post-Hospitalization Medical Expenses? |  | False | False | True | in the context, not in the answer: 60 days / says something is not shown, but the context has: 60 days |
| R03.2 | follow_up | And if I do disclose it, how long before it is covered? | And if I do disclose the pre-existing disease, how long before the pre-existing disease is covered? |  | True | True | True |  |
| R04.2 | follow_up | Is there a similar window if I am late renewing? | Is there a similar window if I am late renewing? |  | False | False | True | not in the context: 30 days / not in the context: not available during the grace period / model refused |
| R05.2 | follow_up | Which plan does that apply to, and how does it work? | Which plan does the Cumulative Bonus (CB) apply to, and how does it work? |  | True | True | True |  |
| R06.2 | follow_up | How early do I have to ask for that? | How early do I have to ask for that? | rewrite adds figures that are not in the user's questions: 12.10 | True | True | True |  |
| R07.2 | follow_up | How soon do I need to tell them about the claim? | How soon do I need to tell the insurance company about the claim? |  | True | True | True |  |
| R08.2 | follow_up | What exactly is that benefit? | What exactly is the benefit for gloves and food charges? |  | False | True | True |  |
| R09.2 | follow_up | What is it, exactly? | What is cashless approval for Home Health Care, exactly? |  | False | False | True | in the context, not in the answer: not available on reimbursement |
| R10.2 | follow_up | Do I get one for buying on the website? | Do I get the Online discount for buying on the website? |  | True | True | True |  |
| R11.2 | follow_up | Can it be taken back if an old claim turns up after I renew? | Can the Cumulative Bonus be taken back if an old claim turns up after I renew? |  | True | True | True |  |
| R12.2 | follow_up | How do I get cashless treatment there? | How do I get cashless treatment for treatment outside India under the Global Health Cover (Emergency Treatments Only)? |  | False | False | True | in the context, not in the answer: network provider |
| R13.2 | follow_up | Can I pay it month by month instead? | Can I pay the grace period for renewing the policy month by month instead? |  | True | True | True |  |
| R14.2 | follow_up | And up to what age can they stay covered as dependants? | And up to what age can dependent children stay covered as dependants? |  | True | True | True |  |
| R15.2 | follow_up | What is that benefit called and how does it work? | What is the Automatic Restore Benefit called and how does it work? |  | True | True | True |  |
| R16.2 | follow_up | What can they do if I don't mention one when I buy the policy? | What can the insurer do if I don't mention a pre-existing condition when I buy the policy? |  | False | True | True |  |
| R17.2 | follow_up | After how many years can they no longer do that? | After how many years can they no longer Incorporate an additional Waiting Period? |  | True | False | True | not in the context: eight continuous years / model refused |
| R18.2 | follow_up | Once they have all my documents, how long do they have to pay? | Once the insurer has all my documents, how long does the insurer have to pay? |  | True | True | True |  |
| R19.2 | follow_up | Do I still get it if I am moved to the ICU? | Do I still get the Daily Cash for Shared Room benefit if I am moved to the ICU? |  | True | True | True |  |
| R20.2 | follow_up | Which of those expenses are not paid? | Which of Pre-Hospitalization and ... expenses are not paid? |  | False | False | True | in the context, not in the answer: transportation or preservation / in the context, not in the answer: organ harvesting |
| R21.2 | follow_up | Are there covers it doesn't apply to? | Are there Individual Policies or Family Policies the Aggregate Deductible doesn't apply to? |  | True | False | True | in the context, not in the answer: e opinion / in the context, not in the answer: preventive health / in the context, not in the answer: global health cover / in the context, not in the answer: overseas travel secure / says something is not shown, but the context has: e opinion, preventive health, global health cover, overseas travel secure |
| R22.2 | follow_up | Which illnesses qualify for that? | Which illnesses qualify for the E-Opinion for Critical Illness? |  | True | True | True |  |
| R23.2 | follow_up | What's the shortest stay that counts? | What's the shortest stay that counts for ICU expenses? |  | False | False | True | in the context, not in the answer: 24 consecutive hours / in the context, not in the answer: day care / model refused / says something is not shown, but the context has: 24 consecutive hours, day care |
| R24.2 | follow_up | After I send everything, when do I get my money? | After I send everything, when do I get my money? |  | False | False | True | not in the context: 30 days / refused by the relevance gate |
| R25.2 | follow_up | If I pay myself and claim later, what extra documents do I need for that? | If I pay myself and claim later, what extra documents do I need for that? | rewrite copies an earlier answer: 'for the global health cover emergency treatments only ...' | False | False | True | in the context, not in the answer: passport / in the context, not in the answer: flight tickets / in the context, not in the answer: accommodation invoices / in the context, not in the answer: written advice / model refused / says something is not shown, but the context has: passport, flight tickets, accommodation invoices, written advice |
| R26.2 | follow_up | How is that different from the emergency-only version? | How is the Global Health Cover (Emergency & Planned Treatments) different from the emergency-only version? |  | False | True | True |  |
| R27.2 | follow_up | What does the emergency-only one exclude? | What does the emergency-only one exclude? |  | True | True | True |  |
| R28.2 | follow_up | Is GST included in that figure? | Is GST included in the annual premium for Optima Secure or Optima Super Secure? |  | False | True | True |  |
| R29.2 | follow_up | Can I get a tax deduction on what I pay? | Can I get a tax deduction on what I pay? |  | True | True | True | cites section numbers not in the context: 80 |
| R30.2 | follow_up | Does it pay for the ambulance to get there? | Does the Hospitalization Expenses pay for the ambulance to get there? |  | True | True | True |  |
| R31.2 | after_refusal | OK, and for Optima Secure with Rs. 10 lakh cover, same age, in Mumbai? | What is the annual premium for Optima Secure with Rs. 10 lakh cover for a 30-year-old in Mumbai? |  | False | True | True |  |
| R32.2 | after_refusal | What about getting treated abroad, then? | Does the policy cover emergency medical treatments outside India? |  | True | True | True |  |
| R33.2 | self_contained | Is there a discount for NRI policyholders? | Is there a discount for NRI policyholders? |  | True | True | True |  |
| R34.2 | self_contained | So how long is a cashless pre-authorization valid? | So how long is a cashless pre-authorization valid? | rewrite names nothing specific | True | True | True |  |
| R35.2 | self_contained | Can they refuse to renew my policy just because I made claims? | Can the insurer refuse to renew my policy just because I made claims? |  | True | True | True |  |
| R36.2 | self_contained | What's the highest medical underwriting loading the insurer can apply? | What's the highest medical underwriting loading the insurer can apply? |  | True | True | True |  |
| R37.2 | self_contained | Does the 30-day waiting period still apply if I've been continuously insured for over a year? | Does the 30-day waiting period still apply if I've been continuously insured for over a year? |  | True | True | True |  |
| R38.2 | self_contained | Can my 4-year-old daughter be insured without a parent on the policy? | Can my 4-year-old daughter be insured without a parent on the policy? |  | False | False | True | figures not in the context: 4 years |
| R39.2 | topic_switch | Which cities are classified as Tier 1 for premium calculation? | Which cities are classified as Tier 1 for premium calculation? |  | True | True | True |  |
| R40.2 | topic_switch | What is domiciliary hospitalization and when is it covered? | What is domiciliary hospitalization and when is it covered? |  | True | True | True |  |
| R41.2 | topic_switch | What is the Preventive Health Check-up benefit? | What is the Preventive Health Check-up benefit? |  | True | True | True |  |
| R42.2 | topic_switch | Can I claim the cost of new spectacles? | Can I claim the cost of new spectacles? |  | True | True | True |  |
| R43.2 | topic_switch | What add-on covers are offered with my:Optima Secure? | What add-on covers are offered with my:Optima Secure? |  | True | True | True |  |
| R44.2 | topic_switch | What limits and conditions apply to claims under the Global Health Cover? | What limits and conditions apply to claims under the Global Health Cover? |  | True | True | True |  |
