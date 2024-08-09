get_ggplot_theme = function(){
  mytheme =
    theme_minimal() + 
    theme(
      text = element_text("Avenir Next Condensed",size=15),
      legend.text = element_text(size=15),
      axis.text = element_text(size=15),
      strip.text = element_text(face="bold",size=15),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      # plot.margin = unit(c(0,0.75,0,0), "cm"),
      legend.box = "vertical",
      legend.position = "top",
      plot.title = element_text(hjust=0.5)
    )
  return(mytheme)
}
