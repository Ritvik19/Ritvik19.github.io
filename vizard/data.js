const nav_data = [
  "Instantiate Vizard",
  "Exploratory Data Analysis",
  "Target Column Analysis",
  "Univariate Analysis",
  "Bivariate Analysis",
  "Trivariate Analysis",
  "Correlation Analysis",
];

const examples = [
  [
    "Classification Case",
    "https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Classification%20Case.ipynb",
  ],
  [
    "Regression Case",
    "https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Regression%20Case.ipynb",
  ],
  [
    "Text Classification Case",
    "https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Text%20Classification%20Case.ipynb",
  ],
  [
    "Unsupervised Case",
    "https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Unsupervised%20Case.ipynb",
  ],
  [
    "Classification Case (Interactive)",
    "https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Classification%20Interactive%20Case.ipynb",
  ],
  [
    "Regression Case (Interactive)",
    "https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Regression%20Interactive%20Case.ipynb",
  ],
  [
    "Unsupervised Case (Interactive)",
    "https://nbviewer.jupyter.org/github/Ritvik19/vizard-doc/blob/main/usage/Unsupervised%20Interactive%20Case.ipynb",
  ],
];

const usage = [
  {
    title: "Instantiate Vizard",
    content: [
      {
        type: "p",
        text: "The Vizard or VizardIn object holds the <strong>DataFrame</strong> along with its configurations",
      },
      {
        type: "code",
        text: `import vizard

class config:
    PROBLEM_TYPE = 'regression' or 'classification' or 'unsupervised'
    DEPENDENT_VARIABLE = 'target_variable'
    CATEGORICAL_INDEPENDENT_VARIABLES = [categorical_features]
    CONTINUOUS_INDEPENDENT_VARIABLES = [continuous features]
    TEXT_VARIABLES = [text features]

viz = vizard.Vizard(df, config)
# for interactive plots use:
viz = vizard.VizardIn(df, config)`,
      },
    ],
  },
  {
    title: "Exploratory Data Analysis",
    content: [
      {
        type: "p",
        text: "After Instatiating the <strong>Vizard</strong> object, you can try different plots for EDA <br> Check Missing Values:",
      },
      {
        type: "p",
        text: "Check Missing Values:",
      },
      {
        type: "code",
        text: `viz.check_missing()`,
      },
      {
        type: "p",
        text: "Count of Unique Values:",
      },
      {
        type: "code",
        text: `viz.count_unique()`,
      },
      {
        type: "p",
        text: "Count of Missing Values by Group:",
      },
      {
        type: "code",
        text: `viz.count_missing_by_group(class_variable)`,
      },
      {
        type: "p",
        text: "Count of Unique Values by Group:",
      },
      {
        type: "code",
        text: `viz.count_unique_by_group(class_variable)`,
      },
    ],
  },
  {
    title: "Target Column Analysis",
    content: [
      {
        type: "p",
        text: "Based on the type of problem, perform a univariate analysis of target column",
      },
      {
        type: "code",
        text: `viz.dependent_variable()`,
      },
    ],
  },
  {
    title: "Univariate Analysis",
    content: [
      {
        type: "p",
        text: "Based on the type of problem, preform univariate analysis of all feature columns with respect to the target column",
      },
      {
        type: "p",
        text: "Categorical Variables:",
      },
      {
        type: "code",
        text: `viz.categorical_variables()`,
      },
      {
        type: "p",
        text: "Continuous Variables:",
      },
      {
        type: "code",
        text: `viz.continuous_variables()`,
      },
      {
        type: "p",
        text: "Text Variables:",
      },
      {
        type: "code",
        text: `viz.wordcloud()

viz.wordcloud_by_group()

viz.wordcloud_freq()`,
      },
    ],
  },
  {
    title: "Bivariate Analysis",
    content: [
      {
        type: "p",
        text: "Based on the type of variables, perform bivariate analysis on all the feature columns",
      },
      {
        type: "p",
        text: "Pairwise Scatter (Continuous vs Continuous):",
      },
      {
        type: "code",
        text: `viz.pairwise_scatter()`,
      },
      {
        type: "p",
        text: "Pairwise Violin (Continuous vs Categorical):",
      },
      {
        type: "code",
        text: `viz.pairwise_violin()`,
      },
      {
        type: "p",
        text: "Pairwise Cross Tabs (Categorical vs Categorical):",
      },
      {
        type: "code",
        text: `viz.pairwise_crosstabs()`,
      },
    ],
  },

  {
    title: "Trivariate Analysis",
    content: [
      {
        type: "p",
        text: "Based on the type of variables, perform trivariate analysis on any of the feature columns",
      },
      {
        type: "p",
        text: "Trivariate Bubble (Continuous vs Continuous vs Continuous):",
      },
      {
        type: "code",
        text: `viz.trivariate_bubble(x, y, s)`,
      },
      {
        type: "p",
        text: "Trivariate Scatter (Continuous vs Continuous vs Categorical):",
      },
      {
        type: "code",
        text: `viz.trivariate_scatter(x, y, c)`,
      },
      {
        type: "p",
        text: "Trivariate Violin (Categorical vs Continuous vs Categorical):",
      },
      {
        type: "code",
        text: `viz.trivariate_violin(x, y, c)`,
      },
    ],
  },
  {
    title: "Correlation Analysis",
    content: [
      {
        type: "p",
        text: "Based on the type of variables, perform correaltion analysis on all the feature columns",
      },
      {
        type: "p",
        text: "Correlation Plot:",
      },
      {
        type: "code",
        text: `viz.corr_plot()`,
      },
      {
        type: "p",
        text: "Chi Square Plot:",
      },
      {
        type: "code",
        text: `viz.chi_sq_plot()`,
      },
    ],
  },
];
