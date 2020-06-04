df <- read.csv("data/AmazonSales.csv", TRUE, ",")
df$Price <- strtoi(df$Price, base = 0L)
#df[c("x", "y")] <- lapply(df[c("x", "y")], function(x) as.numeric(gsub(",", "", x)))
#colMax <- function(df) sapply(df, max, na.rm = TRUE)
#as.numeric(gsub(",", "", df))
#print.data.frame(df)

#countries <- df %>% pull(Country)
#length(unique(countries))

#group_by(.data, Product, add = FALSE, .drop = group_by_drop_default(.data))

#paymentType <- subset(df, Payment_Type == "Mastercard")
#sum(paymentType$Price, na.rm = FALSE)               

#plot(df$Product,df$Price)

tapply(df$Price, df$Product, FUN=sum)

