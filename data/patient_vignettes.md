# Patient Vignettes — AMR-Steward Demo Cases

Plain-English observation texts the model sees during training and demo.
These illustrate the range of case complexity across curriculum levels.

---

## Level 1 Cases (Simple — susceptible organisms, normal renal function)

### Vignette 1
```
Patient: 54-year-old male, admitted from the emergency department.
Infection site: Bacteremia.
Culture result: Klebsiella pneumoniae isolated from blood cultures x2.
Resistance phenotype: Susceptible to standard agents.
Renal function: CrCl 85 mL/min (normal).
Allergies: None reported.
Available antibiogram data: meropenem (MIC 0.25), ceftriaxone (MIC 0.12), ceftazidime-avibactam (MIC 0.5).
```

### Vignette 2
```
Patient: 72-year-old female, nursing home resident admitted for altered mental status.
Infection site: Urinary tract infection (complicated pyelonephritis).
Culture result: Escherichia coli isolated from urine culture.
Resistance phenotype: Susceptible to standard agents.
Renal function: CrCl 70 mL/min (normal).
Allergies: None reported.
Available antibiogram data: ceftriaxone (MIC 0.06), meropenem (MIC 0.12), piperacillin-tazobactam (MIC 4.0).
```

### Vignette 3
```
Patient: 41-year-old male, post-operative day 2 following appendectomy.
Infection site: Intra-abdominal (secondary peritonitis).
Culture result: Escherichia coli isolated from intraoperative cultures.
Resistance phenotype: Susceptible to standard agents.
Renal function: CrCl 95 mL/min (normal).
Allergies: None reported.
Available antibiogram data: ceftriaxone (MIC 0.12), piperacillin-tazobactam (MIC 2.0), ertapenem (MIC 0.06).
```

---

## Level 2 Cases (Moderate — resistant organisms, mild-to-moderate renal impairment)

### Vignette 4
```
Patient: 67-year-old female, admitted to the ICU on post-operative day 3 following colectomy.
Infection site: Bacteremia (central-line associated).
Culture result: Klebsiella pneumoniae isolated from blood cultures x2.
Resistance phenotype: Carbapenem-resistant (CRE). MIC meropenem = 8.0 mg/L.
Renal function: CrCl 35 mL/min (moderate impairment).
Allergies: None reported.
Prior antibiotics: Received cefazolin perioperatively.
Available antibiogram data: meropenem (MIC 8.0), ceftazidime-avibactam (MIC 1.0), colistin (MIC 1.0).
```

### Vignette 5
```
Patient: 58-year-old male, hospital-acquired pneumonia on day 7 of ICU stay (mechanically ventilated).
Infection site: Pneumonia (ventilator-associated).
Culture result: Pseudomonas aeruginosa isolated from bronchoalveolar lavage.
Resistance phenotype: Susceptible to antipseudomonal agents.
Renal function: CrCl 45 mL/min (moderate impairment).
Allergies: Penicillin (reported rash as a child).
Available antibiogram data: cefepime (MIC 2.0), piperacillin-tazobactam (MIC 4.0), meropenem (MIC 0.5), ciprofloxacin (MIC 0.25).
```

### Vignette 6
```
Patient: 63-year-old female, dialysis-dependent ESRD, admitted with fever and rigors.
Infection site: Bacteremia (suspected catheter source).
Culture result: Staphylococcus aureus isolated from blood cultures x3.
Resistance phenotype: MRSA (methicillin-resistant). Vancomycin MIC 1.0 mg/L.
Renal function: CrCl 12 mL/min (severe impairment, non-dialysis days assessed).
Allergies: None reported.
Available antibiogram data: cefazolin (MIC 32.0), vancomycin (MIC 1.0), daptomycin (MIC 0.5), linezolid (MIC 2.0).
```

---

## Level 3 Cases (Complex — MDR organisms, severe renal impairment, allergy constraints)

### Vignette 7
```
Patient: 78-year-old male, liver transplant recipient on immunosuppression, admitted from outpatient clinic with sepsis.
Infection site: Bacteremia.
Culture result: Pseudomonas aeruginosa isolated from blood cultures x2.
Resistance phenotype: MDR — resistant to cefepime, piperacillin-tazobactam, and meropenem.
Renal function: CrCl 18 mL/min (severe impairment).
Allergies: Penicillin (anaphylaxis), cephalosporin (hives).
Available antibiogram data: cefepime (MIC 64.0), meropenem (MIC 32.0), ceftazidime-avibactam (MIC 16.0), cefiderocol (MIC 1.0), colistin (MIC 2.0).
```

### Vignette 8
```
Patient: 85-year-old female, long-term care facility resident with indwelling urinary catheter and recent broad-spectrum antibiotic exposure.
Infection site: Urinary tract infection (complicated, catheter-associated).
Culture result: Escherichia coli isolated from urine culture.
Resistance phenotype: MDR (CRE) — carbapenem MIC = 16.0 mg/L.
Renal function: CrCl 22 mL/min (severe impairment).
Allergies: Sulfonamide (Stevens-Johnson syndrome), vancomycin (flushing, possibly Red Man Syndrome).
Available antibiogram data: meropenem (MIC 16.0), ceftazidime-avibactam (MIC 2.0), meropenem-vaborbactam (MIC 1.0), cefiderocol (MIC 0.5).
```

### Vignette 9
```
Patient: 49-year-old male, neutropenic post-induction chemotherapy for AML, day 12 of neutropenia.
Infection site: Bacteremia (suspected gut translocation).
Culture result: Enterococcus faecium isolated from blood cultures x2.
Resistance phenotype: VRE (vancomycin-resistant, ampicillin-resistant).
Renal function: CrCl 38 mL/min (moderate impairment).
Allergies: None reported.
Available antibiogram data: ampicillin (MIC 32.0), vancomycin (MIC 64.0), daptomycin (MIC 1.0), linezolid (MIC 2.0).
```

### Vignette 10
```
Patient: 61-year-old female, post-Whipple procedure, re-admitted day 10 post-op with anastomotic leak and septic shock.
Infection site: Intra-abdominal (tertiary peritonitis, healthcare-associated).
Culture result: Klebsiella pneumoniae (CRE) and Enterococcus faecalis (VSE) isolated from drain cultures.
Resistance phenotype: K. pneumoniae — MDR/CRE; Enterococcus — susceptible.
Renal function: CrCl 28 mL/min (severe impairment).
Allergies: Fluoroquinolone (tendinopathy).
Available antibiogram data: meropenem (MIC 32.0), ceftazidime-avibactam (MIC 1.0), colistin (MIC 1.5), ampicillin (MIC 1.0).
Note: Prescribe for the most resistant organism. Source control (IR drainage) already performed.
```
