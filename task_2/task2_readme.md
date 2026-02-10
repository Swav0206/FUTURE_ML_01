#  Support Ticket Classification & Prioritization  
### Machine Learning Internship – Future Interns (Task 2)

---

##  Project Overview

In real-world IT organizations, support teams receive thousands of tickets daily.  
Manually reading, categorizing, and prioritizing these tickets is time-consuming and error-prone.

This project builds an **automated support ticket classification and prioritization system** using **Natural Language Processing (NLP)** and **Machine Learning**, helping organizations:

- Automatically route tickets to the correct support team
- Identify urgent issues faster
- Reduce response time and support backlog
- Improve overall service efficiency

---

##  Objectives

The system is designed to:

- Read raw IT support ticket text
- Automatically classify tickets into issue categories
- Assign a priority level (High / Medium / Low)
- Provide a dynamic, interactive interface for real-time predictions

---

##  Dataset Used

**IT Service Ticket Classification Dataset**  
File: `all_tickets_processed_improved_v3.csv`

- ~47,000 real IT support tickets
- Clean, text-separable technical categories
- Suitable for NLP-based classification tasks

### Key Columns:
- `Document` → Ticket description (input text)
- `Topic_group` → Ticket category (target label)

---

##  Tech Stack

- **Programming Language:** Python  
- **Development Environment:** VS Code  
- **Libraries Used:**
  - pandas, numpy
  - scikit-learn
  - TF-IDF Vectorizer
  - Logistic Regression

---

##  Methodology

###  Text Preprocessing
- Lowercasing
- Removing punctuation & special characters
- Removing extra whitespace
- Filtering very short or empty tickets

###  Feature Extraction
- TF-IDF Vectorization
- Unigrams + Bigrams
- Stopword removal
- Sparse, high-dimensional text representation

###  Ticket Classification (ML Model)
- **Model:** Logistic Regression
- **Why:** Efficient, interpretable, and highly effective for text classification
- **Class balancing:** Enabled to handle uneven ticket distribution

###  Priority Assignment (Rule-Based)
A rule-based priority engine assigns urgency levels using keywords:

- **High Priority:** outage, down, failed, crash, urgent, not working
- **Medium Priority:** issue, request, problem, delay
- **Low Priority:** how to, information, query, guidance

This hybrid ML + rule-based approach mirrors real IT service workflows.

---

##  Results & Performance

- **Ticket Category Classification Accuracy:** ~84%
- Strong precision and recall across technical categories
- Clear and interpretable confusion matrix
- Reliable generalization on unseen tickets

This performance is considered **production-grade** for automated IT ticket routing systems.

---

##  Dynamic User Interaction

The system supports **real-time dynamic input** via the terminal.

Example:
Enter support ticket text: VPN is down and users cannot access the network urgently

Predicted Category: Network Issue
Assigned Priority: High



This makes the project usable as a real internal support tool.

---

##  Business Impact

This system can help organizations:

- Reduce manual ticket triaging
- Prioritize critical incidents instantly
- Improve SLA compliance
- Enhance customer and employee satisfaction
- Scale IT support operations efficiently

---

##  Conclusion

This project demonstrates a complete, end-to-end NLP-based decision support system for IT service management.  
By combining Machine Learning with explainable business rules, it delivers both **accuracy** and **operational transparency**.

---

##  Author

**Sweeti Rathore**  
Machine Learning Intern – Future Interns  

---
