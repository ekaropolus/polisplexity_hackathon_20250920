# Polisplexity: Internet of AI Cities (Hackathon Edition)

[cite_start]Welcome to the hackathon edition of Polisplexity, a platform designed to transform complex urban data into clear, actionable answers with a single click[cite: 133]. [cite_start]Our mission is to solve the problem of slow, political, and underutilized urban decision-making by providing businesses and city planners with fast, trustworthy insights[cite: 105, 106].

[cite_start]This repository contains a core subset of the main Polisplexity platform, specifically for **The Internet of Agents Hackathon**[cite: 1].

## The Problem

Cities are complex systems. [cite_start]Data is often messy and siloed, making it difficult for businesses to identify optimal locations or for governments to make efficient infrastructure decisions[cite: 106]. [cite_start]This leads to wasted resources and missed opportunities[cite: 105].

## Our Solution: The Internet of AI Agents

[cite_start]Polisplexity tackles this with an **Internet of AI Agents for Cities**[cite: 111]. [cite_start]We use a messaging-first approach where specialized AI agents, built on the Coral Protocol, collaborate to analyze data and generate knowledge[cite: 2, 114].

This hackathon project provides you with the key Django modules to interact with this system:

* **`pxy_sites`**: The core location intelligence agent. [cite_start]You give it a place and parameters, and it computes site viability scores using OpenRouteService for isochrones, returning JSON data and map previews (PNG/GeoJSON)[cite: 88].
* **`pxy_sami`**: The scientific backbone. [cite_start]This agent performs urban scaling analytics based on established scientific models to compare cities and regions fairly by size[cite: 22, 89].
* **`pxy_agents_coral`**: Your gateway to the larger "Internet of Agents." [cite_start]This module acts as a proxy to the Coral Protocol, formatting requests and normalizing JSON responses for any frontend application[cite: 99].

## Technology Stack

[cite_start]This project is a **Django Monolith** [cite: 57] supported by a robust set of technologies including:

* [cite_start]**Web Server**: Gunicorn [cite: 57] + [cite_start]Traefik (as a reverse proxy) [cite: 54]
* [cite_start]**Async Tasks**: Celery [cite: 58]
* [cite_start]**Datastores**: Postgres for primary data, Redis for caching and message brokering [cite: 81, 83]
* [cite_start]**Key Services**: Integrations with OpenAI [cite: 53][cite_start], OpenRouteService [cite: 51][cite_start], and Meta Webhooks (WhatsApp/Facebook) [cite: 50]

## Getting Started

1.  **Clone the Repository**:
    ```bash
    git clone [your-github-repo-url]
    cd [your-repo-name]
    ```
2.  **Set Up Environment**:
    * Create and activate a Python virtual environment.
    * Install dependencies: `pip install -r requirements.txt`
3.  **Configure**:
    * Copy the `.env.example` file to a new file named `.env`.
    * Fill in the necessary environment variables (API keys, database URL, etc.) in the `.env` file.
4.  **Run the App**:
    ```bash
    python manage.py migrate
    python manage.py runserver
    ```

## License & Usage

This project is source-available and free to use for the **Internet of Agents Hackathon**. You are encouraged to learn from it, build upon it, and use it to create amazing things during this event.

However, it is **not** under a traditional open-source license. It is provided without warranty and is not licensed for unrestricted public redistribution or commercial use outside of this hackathon. For a formal license, this work can be considered under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**.

---

> ### A Note from the Architects
>
> We are the builders, the architects, and the scientists who poured countless hours of research, code, and expertise into making Polisplexity a reality. [cite_start]This project stands on a deep foundation of urban science, data engineering, and AI development[cite: 10].
>
> We are sharing this slice of our work in good faith for the collaborative spirit of this hackathon. Please honor that spirit. Use this code to learn and to create something new. **Do not simply rebrand our work and present it as your own.**
>
> Give credit where it is due. True genius is about creation and collaboration, not appropriation. Let's build the future together.

---
