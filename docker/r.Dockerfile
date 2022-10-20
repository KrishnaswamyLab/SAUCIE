FROM r-base:4.0.2

RUN R -e "install.packages('packrat', repos='http://cran.r-project.org')"
