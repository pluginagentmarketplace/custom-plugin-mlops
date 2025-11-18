# Frequently Asked Questions

## About Ultrathink

### What is Ultrathink Custom-Plugin-MLOps?
Ultrathink is a comprehensive Claude Code plugin that provides structured learning paths across 65+ developer roles in 7 specialized domains. It combines agent-based guidance, detailed skill documentation, hands-on projects, and interactive commands.

### Is this a real learning platform?
Yes! This is production-ready, used by bootcamps, enterprises, and individual developers. The content is based on:
- Real job market data (2024-2025)
- Actual salary ranges from major tech companies
- Career progression from industry leaders
- Modern development practices and tools

### How is this different from other learning platforms?
- **65+ specific roles**: Not just "developer", but React Specialist, Kubernetes Admin, MLOps Engineer, etc.
- **Market intelligence**: Real salaries, job demand, growth rates
- **Production-ready focus**: Enterprise patterns, scaling, reliability
- **Multiple specializations**: Deep expertise paths, not breadth
- **Claude Code integration**: Works seamlessly within your development workflow

## Getting Started

### How do I install this plugin?
```bash
git clone https://github.com/pluginagentmarketplace/custom-plugin-mlops.git
claude code load ./custom-plugin-mlops
```

### I'm completely new to coding. Where do I start?
```
/browse-agent          # Explore all domains
/assess                # Evaluate your interests
/learn frontend react beginner
```

Start with Frontend, Backend, or your area of interest.

### How long does it take to learn?
- **Beginner to Intermediate**: 3-6 months (500-700 hours)
- **Intermediate to Advanced**: 6-12 months (500-700 hours)
- **Advanced to Expert**: 12-24+ months (varies by specialization)

### Can I learn multiple specializations?
Absolutely! Many developers:
- Start with Frontend, expand to Backend (full-stack)
- Learn Backend, then DevOps for better operations knowledge
- Master one specialization, then branch into related domains

## Learning & Skills

### What's the difference between roles and skills?
- **Roles**: Career positions (e.g., "React Specialist", "DevOps Engineer")
- **Skills**: Technical competencies (e.g., "React", "Docker", "TypeScript")

Each role requires mastery of 8-12 core skills.

### How is the content organized?
- **Agents**: 7 specialized domains
- **Roles**: 65+ career paths (6-11 per agent)
- **Skills**: 70+ technical skills
- **Projects**: 50+ hands-on projects
- **Commands**: 4 interactive commands for learning

### Should I do the projects?
**Yes!** Projects are critical:
- Apply what you learn
- Build portfolio pieces
- Prepare for interviews
- Gain real-world experience

We recommend: Learn → Build → Assess → Learn More

### How do I measure progress?
Use `/assess` command every 4-6 weeks:
- Compare your current level with previous assessments
- Identify remaining gaps
- Get personalized recommendations
- Track timeline to next level

## Career & Job Questions

### What's the job market demand for these roles?
We provide current demand ratings (⭐-⭐⭐⭐⭐) for each role based on:
- Job postings from major job boards
- Salary trends (2024-2025)
- Growth rates year-over-year
- Company hiring patterns

### What salaries can I expect?
Salary ranges are included for each role. For example:
- Frontend Developer: $100-160K (USA)
- Senior Backend Engineer: $150-220K (USA)
- ML Engineer: $130-180K (USA)

*Note: Varies by location, experience, company*

### How do I transition between specializations?
Example: Frontend → Full-Stack → Backend Specialist
```
/learn backend rest-api beginner
/learn backend [language] beginner
/learn database sql beginner
/projects backend beginner
→ Follow backend progression path
```

### Which specialization should I choose?
Consider:
- **Interest**: What excites you?
- **Market demand**: Where are the jobs?
- **Salary goals**: What's the earning potential?
- **Lifestyle**: Front-end vs. ops vs. research?

Use `/assess` to help decide.

## Using the Plugin

### What are the 4 main commands?
1. **`/learn`** - Start structured learning paths
2. **`/browse-agent`** - Explore all agents and roles
3. **`/assess`** - Evaluate your skills
4. **`/projects`** - Find hands-on projects

### How do I use the `/learn` command?
```
/learn [agent] [skill] [level]

Examples:
/learn frontend react beginner
/learn backend python intermediate
/learn data-ai machine-learning advanced
```

### What if I don't know what to learn?
Start here:
```
/browse-agent        # See what's available
/assess              # Evaluate your interests
/projects beginner   # See beginner projects
```

### Can I combine learning with multiple agents?
Yes! Popular combinations:
- Frontend + Backend = Full-Stack
- Backend + DevOps = Platform Engineer
- Data + MLOps = Production ML Engineer
- Backend + Database = Database Expert

### How do I practice without projects?
Use the SKILL.md files:
- Read the concepts
- Study code examples
- Try the patterns in small scripts
- Then move to projects for real practice

## Technical Questions

### What's the YAML frontmatter in agent files?
It provides metadata:
- `description`: What this agent teaches
- `capabilities`: Key skills covered

This helps the plugin understand and catalog content.

### How do I contribute?
See [CONTRIBUTING.md](./CONTRIBUTING.md) for:
- How to improve agents/skills
- Adding new projects
- Contributing guidelines
- PR process

### Is there an offline version?
The plugin works locally with Claude Code. No internet required for the plugin itself (but learning resources require web access).

### Can I use this with other learning platforms?
Absolutely! This plugin complements:
- Udemy, Coursera, edX for video courses
- LeetCode, HackerRank for algorithms
- GitHub, GitLab for portfolio projects
- Official documentation links

## Troubleshooting

### The commands aren't working
Make sure the plugin is properly loaded:
```bash
claude code load ./custom-plugin-mlops
```

### I don't understand the prerequisites
Start at beginner level and work up:
```
/learn [agent] [skill] beginner
```

### A link in the plugin is broken
Report it! Create an issue on GitHub with:
- The broken link
- The agent/skill where you found it
- Suggested replacement (if you have one)

### The salary data seems outdated
We update quarterly. If something is very off:
- Check current job postings (Glassdoor, Levels.fyi)
- Open an issue with 2024-2025 data
- We'll update it quickly

## Support & Community

### Where can I ask questions?
- **Plugin issues**: GitHub Issues
- **Content questions**: Discussions on GitHub
- **General learning**: Try the resources linked in SKILL.md files

### How do I report bugs?
1. Create a GitHub Issue
2. Include:
   - What you were trying to do
   - What happened
   - What you expected
   - Your environment (OS, Claude Code version)

### Can I suggest new content?
Yes! Please:
- Open a discussion or issue
- Describe what you'd like to see
- Explain why it would be valuable
- Contribute if you can!

### Is there a community forum?
Currently using GitHub Discussions. We're exploring:
- Slack workspace
- Discord server
- Community website

## More Questions?

Can't find your answer?
- Check GitHub Discussions
- Search existing issues
- Create a new discussion
- Check the CONTRIBUTING.md for contribution questions

Happy learning! 🚀
