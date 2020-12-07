library(dplyr)
library(caret)
library(randomForest)
library(eegkit)
library(rlist)
library(edfReader)
library(eegUtils)
library(e1071)
library(data.table)
library(ggplot2)
library(effsize)

#0. Abgleichen der EEG-Daten und Testergebnisse. Zur Identifizierung von Probanden, die sowohl den Test als auch das EEG abgeschlossen haben.

#0.1 Setzung des Working directories mit den EEG-Daten (.EDF-Dateien)
setwd("/Volumes/Ext_Fil/Hs_Aalen/Vorlesungen/1. Semester/Data_Mining/WS1920/Vorlesungen/BDI/EEG_EDF")

#0.2 Erstellung einer Liste mit den EEG-Daten pro Proband
eegFiles = list.files("/Volumes/Ext_Fil/Hs Aalen/Vorlesungen/1. Semester/Data_Mining/Vorlesungen/BDI/EEG_EDF")

#0.3 Setzung des Working directories mit den Testergebnissen (CSV-Datei)
setwd("/Volumes/Ext_Fil/Hs Aalen/Vorlesungen/1. Semester/Data_Mining/Vorlesungen/BDI/CSV")

#0.4 Einlesen der Testergebnisse
bdiTestResults <- read.csv("/Volumes/Ext_Fil/Hs Aalen/Vorlesungen/1. Semester/Data_Mining/Vorlesungen/BDI/CSV/BDI.csv", head = TRUE, sep=";", na.strings = c("","NA","N/A", "n/a"))

#0.5 Bereinigung der BDI Testergebnisse.(Entfernung von Probanden mit fehlenden Werten (N/A)) Ergebnis: 210 Probanden
bdiTestResults <- bdiTestResults[complete.cases(bdiTestResults), ]

#0.6 Auswahl der Spalten mit der ProbandenID und dem Endergebnis des Tests. (Zur Identifizierung der jeweiligen Probanden und zur späteren Bildung der Labels mit den Testergebnissen (0 = Keine depressiven Tendenzen, 1 = Depressive Tendenzen))
bdiTestResults <- select(bdiTestResults, participant_id, BDI_summary_sum)

#0.7 Auswahl der Extremwerte unter den Probanden (Maximale Extremwerte = Alle Probanden mit einem Testergebnis von >11 (22 Probanden) und alle Probanden mit einem Testergebnis von <2 (21 Probanden))
bdiTestResultsExtremes <- bdiTestResults[(bdiTestResults$BDI_summary_sum < 2 | bdiTestResults$BDI_summary_sum > 10),]
bdiTestResultsLastProband <- bdiTestResults[which((bdiTestResults$BDI_summary_sum ==2)),]
bdiTestResultsLastProband <- bdiTestResultsLastProband[1:1,]
bdiTestResultsExtremes <- rbind(bdiTestResultsExtremes, bdiTestResultsLastProband)

#0.8 Eine Zwischenvarable wird generiert, um die Spalte "participant_ID" der Tabelle bdiTestResults mit der Endung ".edf" zu versehen.
probandIdentification <- paste0(bdiTestResultsExtremes$participant_id , ".edf")

#0.9 Dies wird im Anschluss dafür genutzt, um die EEG-Daten der Probanden zu entfernen, die den BDI-Test nicht vollständig absolviert haben. (=44 Probanden bleiben übrig)
eegFiles <- intersect(eegFiles, probandIdentification)

#1: Durchführung des Signal Preprocessing. Beinhaltet die Durchführung von Low-Pass and High-Pass Filtern, ICA und FFT. 

#Eine For-Schleife wird genutzt, um die Namen der .EDF-Dateien in der Variablen "eegFiles" nacheinander abzurufen, um mit diesen die EEG-Daten einzeln abzurufen.
for(j in 1:length(eegFiles)){
  #1.01 Setzung des Working directories mit den EEG-Daten (.EDF-Dateien)
  setwd("/Volumes/Ext_Fil/Hs_Aalen/Vorlesungen/1. Semester/Data_Mining/Vorlesungen/BDI/EEG_EDF")
  #1.02 Generierung der Variable "eegBdiProband". Dieses beinhaltet zunächst einen Substring aus der jeweiligen ProbandenID.(1-10, weil die ID 10 Zeichenlang ist)
  eegBdiProband <- as.character(paste(substr(eegFiles[j],1,10)))
  #1.03 Aus dieser Variable wird nun ein DataFrame erstellt. Dies dient dazu, um mit der "merge"-Funktion zu arbeiten, da diese zwei Dataframes als Eingabewerte benötigt.
  eegBdiProband <- as.data.frame(eegBdiProband)
  #1.04 "merge"-Funktion. Eine Verknüpfung mit dem DataFrame "bdiTestResults" wird anhand der Spalten "eegBdiProband" (des Dataframes "eegBdiProband") und "participant_id" (des Dataframes "bdiTestResults) getätigt.
  #Am Ende steht ein DataFrame mit dem Probandennamen und dessen Testergebnis. Dient später zum Labeln.
  eegBdiProband <- merge(eegBdiProband, bdiTestResults, by.x = "eegBdiProband", by.y = "participant_id", all.x = TRUE)
  
  #1.1. Importieren der rohen EEG (.EDF)-Dateien. (Funktionalität: eegUtils)
  eegImport <- import_raw(eegFiles[j])
  
  #1.2. Entfernen des VEOG-Channels, um Augenblink-Artefakte zu entfernen. (Funktionalität: eegUtils)
  eegImport <- select_elecs(eegImport, electrode = "VEOG", keep = FALSE)
  
  #1.3. toupper um die Channel-Namen der Signale in Großbuchstaben umzuwandeln. 
  #(Ansonsten funktioniert das Hinzufügen der Elektroden-Locations nicht, da die Funktion "electrode_location" mit den Soaltennamen matchen muss) 
  colnames(eegImport$signals) <- toupper(colnames(eegImport$signals))
  
  #1.4 Hinzufügen der Channel-Locations. (Funktionalität: eegUtils)
  eegImport <- electrode_locations(eegImport)
  
  #1.5 Filtern der Signale (Band-Pass mit 0.5-50HZ)  (Funktionalität: eegUtils)
  eegImport <- eeg_filter(eegImport, low_freq = 0.5, high_freq = 50, method = "fir")
  
  #1.6 Hinzufügen der Epochen-Codes. (Augen geschlossen / Augen geöffnet / Augen offen & geschlossen) (Funktionalität: eegUtils)
  #Epochs eyes-open
  eyesopen <- epoch_data(eegImport, events = c(eegImport[["events"]][["event_type"]][5]))
  
  #1.7 Durchführung der ICA auf allen Epochen. (Funktionalität: eegUtils)
  #eyes-open
  icaopen <- ar_FASTER(eyesopen)
  
  #2: Spectralanalyse
  #2.1: Durchführung der Fourier-Transformation (Funktion: eegfft)
  eegFft <- eegfft(icaopen[["signals"]], 250, 0.5, 50) 
  #2.2.: Abspeichern der Frequency und Strength in einem Dataframe  
  Frequency <- eegFft$frequency
  Strength <- eegFft$strength #Vll eher Channels nennen?
  powerSpectralDensity <- data.frame(Frequency, Strength) #Spektraleleichtungsdichte mit der Frequenz 
  
  #3: Frontal Asymmetry  
  #3.1: Entfernung aller Channel, die nicht zur Messeung der Frontal Asymmetry benötigt werden. (Alle Channel außer F3/F4 % F7/F8)
  #Entfernen der Channel, die nicht zur Asym. gehören. (Erstmal ohne die Summe zu divideren)
#  powerSpectralDensity <- powerSpectralDensity[,grep(paste(frontalAsymmetry, collapse = "|"), colnames(powerSpectralDensity))]
  
  #3.2. Vergabe eines einheitlichen Namens der Frequency.
  names(powerSpectralDensity)[names(powerSpectralDensity) == 'eegFft.frequency'] <- 'Strength'
  
  #3.4. Jeder Proband erhält ein Label, um diesen zu identifizieren. 
  #0 = Keine depressiven Tendenzen, d.h. BDI-Testergebnis < 11
  #1 = Depressive Tendenzen, d.h. BDI-Testergebnis > 10
  labelProband <- NULL
  if (eegBdiProband$BDI_summary_sum > 10){
    labelProband <- rbind(labelProband, 1)
  } else{
    labelProband <- rbind(labelProband, 0)   
  }  
  
  identifier = unique(labelProband)
  
  #4: Bestimmung der 99 Frequenzbereiche
  #4.1: Generierung der Variablen für: Die Aufrechnung der Frequenzbereiche ("x" und "y"), 
  #dem Abspeichern des Labels ("labelDepression" und "labelHealthy"),
  #, der Speicherung des jeweiligen Frequenzbereiches (frequencyBands),
  # der Speicherung der jeweiligen Subbänder und
  #die Zusammenfassung der Subbänder, des Labels und des jeweiligen Frequenzbereiches ("dfIdentified")
  x <- 0.5 
  y <- x + 0.5000 #1
  labelDepression <- NULL
  labelHealthy <- NULL
  frequencyBands <- NULL
  dfSubBands <- NULL
  dfIdentified <- NULL
  
  for(i in 1:99){ #23365/99 = 236 Probanden pro Frequenzband
    #4.2: Auswahl der einzelnen Frequenzbänder  
    dfSplit <- subset(powerSpectralDensity, subset = (( powerSpectralDensity$Frequency >= x & powerSpectralDensity$Frequency <= y))) 
    dfSubBands <- rbind(dfSubBands, colSums(dfSplit)[2:ncol(dfSplit)])
    #rowCount = zur Prüfung, ob Subbänder zum Aufteilen bestehen.
    rowCount = nrow(dfSplit) 
    
    if(rowCount != 0){
      
      #4.3: Den Frequenzbereich und das Label an den jeweiligen Probanden anhängen 
      
      if(identifier == "1"){ #Depressive Tendenzen
        labelDepression <- rbind(labelDepression, 1)
        id <- as.character(paste(x, "-", y))
        frequencyBands <- rbind(frequencyBands, id)
        dfIdentified <- cbind(dfSubBands, labelDepression, frequencyBands)
        colnames(dfIdentified) <- c(colnames(dfSubBands), "Depressiv", "frequencyBands") 
        rownames(dfIdentified) <- NULL
      } 
      else {
        labelHealthy <- rbind(labelHealthy, 0) #Keine depressiven Tendenzen
        id <- as.character(paste(x, "-", y))
        frequencyBands <- rbind(frequencyBands, id)
        dfIdentified <- cbind(dfSubBands, labelHealthy, frequencyBands)
        colnames(dfIdentified) <- c(colnames(dfSubBands), "Depressiv", "frequencyBands") #Frequenzbereich No.1 ist 0.5 bis 1 und so hoch
        rownames(dfIdentified) <- NULL
      }
    } else{print(i)}
    
    #Hochzählen des Frequenzbereiches um am Ende auf die 50Hz zu gelangen. Hochzählen erfolgt in 0.5 Inkrementen.
    x <- x+0.5000
    y <- y+0.5000 
  }
  #4.4: Abspeichern der Channel, des Labels und der Frequenzbereiche in einer CSV.
  folder <- as.character(paste("/Volumes/Ext_Fil/Hs Aalen/Vorlesungen/1. Semester/Data_Mining/Vorlesungen/BDI/Probanden_allC/Proband", j)) #Mit j wird immer der Proband hinzugef?gt, 
  #Wenn der Ordner in der Variable "folder" noch nicht existiert, wird dieser erstellt
  if(dir.exists(folder)== FALSE){
    dir.create(folder) 
  }
  wd <- as.character(paste("/Volumes/Ext_Fil/Hs Aalen/Vorlesungen/1. Semester/Data_Mining/Vorlesungen/BDI/Probanden_allC/Proband", j)) #Pro Proband 169 Files 
  setwd(wd)
  #Bezeichnung der .CSV-Datei und erstellen dieser
  file <- as.character(paste("Proband ", j, ".csv", sep="")) 
  write.csv(dfIdentified, file = paste(file))  
}

#5: Einteilung der Channels in 99 Subbänder
#5.1: Generierung der Variablen, zum Speichern folgender Informationen:
#"fileLoad" dient zur Speicherung der Informationen aus der jeweiligen Patienten-CSV
#"dfProband" beherbergt diese Informationen
#"classiferOverall" ist eine Auflistung der Label von allen Probanden
#"probandsOverall" ist eine Aufzählung aller Probanden (Wird nicht benötigt?)
#"list" dient für den Zähler der For-Schleife, um alle Probanden durchzulaufen. Das Intervall 1:44 wurde ausgewählt, aufgrund der Gesamtanzazl der Probanden. (44 Probanden) 
fileLoad <- NULL
dfProband <- NULL 
dfProbandSubbands <- NULL
classifiersOverall <- NULL
probandsOverall <- NULL
list = c(1:44)

#5.2: Einlesen der .CSV-Dateien
for(p in 1:length(list)){
  #Setzung des Working Directories
  wd2 <- as.character(paste("/Volumes/Ext_Fil/Hs Aalen/Vorlesungen/1. Semester/Data_Mining/Vorlesungen/BDI/Probanden_allC/Proband", p))
  setwd(wd2)
  #Einladen der CSV des Probanden
  listingFiles <- list.files(wd2,  pattern = ".csv") 
  
  #Einlesen der einzelnen Zeilen der CSV und einfügen dieser in die Variable fileLoad. Diese werden untereinander eingefügt. Ergebnis: 99 Zeilen in der Variablen.
  fileLoad <- do.call(rbind, lapply(listingFiles, function(x) read.csv(x, stringsAsFactors = TRUE)))
  #Umwandlung der einzelnen Zeilen in ein DataFrame. Entfernen der 1. Spalte, da diese keinen Channel beinhaltet.
  dfProband <- data.frame(fileLoad) 
  dfProband <- dfProband[,-1]
  
  #Hinzufügen des Labels pro Proband
  classifierChar <- dfProband$Depressiv #1 oder 0
  classifier <- unique(classifierChar) #Gibt nur einen 
  classifiersOverall <- rbind(classifiersOverall, classifier)
  
  #Zusammenschluss aller der Channel und dem Label pro Proband in der Variablen "dfProbanden" (Muss noch geändert werden!).
  #Eine Zeile beinhaltet alle Information von genau einem Patienten
  dfProbandSubbands <- rbind(dfProbandSubbands, dfProband)
  #Hinzufügen der Probandennummer
  probandSingle <- as.character(paste("Proband", c(p))) #
  #Liste aller Probanden
  probandsOverall <- rbind(probandsOverall, probandSingle)
  
  #Leeren der Variablen "fileLoad" um in der nächsten Iteration den nächsten Patienten einzuladen
  fileLoad <- NULL
  #Ausgaben in der Konsole. Bestätigung, dass der Proband hinzugefügt wurde und Benachrichtigung zum aktuellen Stand
  print(probandSingle) 
} 

#5.3: For-Schleife zum Aufteilen der Channel
#Übergabe des DataFrames mit allen Channeln und Patienten in eine neue Variable. Diese Variable wird später iterativ geleert,
#um die 99 Zeilen der Probanden in Channel umzuwandeln. Ebenfalls werden all die Spalten entfernt, die zur Einteilung in die Sub-Bänder nicht benötigt werden. 
df_kopie2 <- dfProbandSubbands
#frontalAsymmetry <- c("^F3$", "^F4$","^F7$","^F8$", "^F5$","^F1$",
                      "^F2$","^F6$", "^FP1$", "^FP2$", "^AF3$","^AF4$", 
                      "^AF7$", "^AF8$")

#df_kopie2 <- df_kopie2[,grep(paste(frontalAsymmetry, collapse = "|"), colnames(df_kopie2))]

# df_kopie2$Frequency <- NULL
# df_kopie2$Depressiv <- NULL

#Deklarierung einer neuen Variable, die das Endergebnis der For-Schleife beinhaltet
df_kopie <- NULL

#For-Schleife zur Einteilung in die Sub-Bänder. Von k bis zur maximalen Anzahl der Probanden (44 Probanden)
for (k in 1:length(probandsOverall)){
  tmpProb <- NULL
  #Deklarierung einer Zwischenvariable, die eine umgedrehte Tabelle der Variablen "df_kopie2" beinhaltet. 
  #Die Channel (vor der Einteilung) sind nun Zeilen und die 99 Zeilen (= 99 unterschiedliche Werte pro Channel) sind damit nun Spalten
  tmpProb <- as.data.frame(t(unlist(df_kopie2[1:99,1:14])))
  #Eine Variable "counter" wird erstellt. Diese hilft dabei, jedes Mal auf 99 Zeilen der Übersichtstabelle (in denen sich die Probanden befinden)
  #zu zugreifen. 
  #1:99 Zeilen = 1 Proband, mit je 0.5 Hz Frequenzschritten
  counter <- 1:99
  #1:4, da wir 4 Channeln (F3,F4,F7,F8) haben 
  for (b in 1:14){ 
    #Speichert in einer Variable 99-mal den Channelnamen einen spezifischen Channelnamen ein
    unlistingVariabels <- rep.int(colnames(df_kopie2)[b], 99) 
    #Ändert die Channelnamen in Channelnamen + Frequenzbereich um
    colnames(tmpProb)[counter] <- paste(unlistingVariabels,"_",frequencyBands)  
    #Zählt die Variable "counter" hoch, um den nächsten Probanden auszuwählen
    counter <- counter+99
  }
  #Ergebnisse werden in eine neue Tabelle eingehängt
  df_kopie <- rbind(df_kopie, tmpProb)
  #Die Übersichtstabelle wird um 99 Zeilen reduziert, um die Informationen des nächsten Probanden abzurufen
  df_kopie2 <- df_kopie2[-(1:99),]
}

#6. Berechnung des Frontal Alpha Asymmetry Indexes (FAA)
#6.1 Vorbereitung der For-Schleife
#Die Variable "dfAlphaPower" erhält die Tabelle mit den 99 Subändern pro Channel zur Berechnung des FAAs
for(i in 1:1){
dfAlphaPower <- NULL
dfAlphaPower <- df_kopie
 # dfAlphaPower <- df_kopie[16:25]
 # dfAlphaPower <- cbind(dfAlphaPower, df_kopie[115:124])
 # dfAlphaPower <- cbind(dfAlphaPower, df_kopie[214:223])
 # dfAlphaPower <- cbind(dfAlphaPower, df_kopie[313:322])
 # dfAlphaPower <- cbind(dfAlphaPower, df_kopie[412:421])
 # dfAlphaPower <- cbind(dfAlphaPower, df_kopie[511:520])
 # dfAlphaPower <- cbind(dfAlphaPower, df_kopie[610:619])
 # dfAlphaPower <- cbind(dfAlphaPower, df_kopie[709:718])
 # dfAlphaPower <- cbind(dfAlphaPower, df_kopie[808:817])
 # dfAlphaPower <- cbind(dfAlphaPower, df_kopie[907:916])
 # dfAlphaPower <- cbind(dfAlphaPower, df_kopie[1006:1015])
 # dfAlphaPower <- cbind(dfAlphaPower, df_kopie[1105:1114])
 # dfAlphaPower <- cbind(dfAlphaPower, df_kopie[1204:1213])
 # dfAlphaPower <- cbind(dfAlphaPower, df_kopie[1303:1312])
}
 
#Deklarierung von Variablen, die subsets der Channel beherbergen. (Wird dafür benötigt, um für jeden Frequenzbereich den FAA zu berechnen)

alphaPowerF8F7Split <- NULL
alphaPowerF4F3Split <- NULL
alphaPowerAF4AF3Split <- NULL
alphaPowerAF8AF7Split <- NULL
alphaPowerF6F5Split <- NULL
alphaPowerFP2FP1Split <- NULL
alphaPowerF2F1Split <- NULL

p <- 0.5
l <- p + 0.5
dfAlpha <- NULL
dfAlpha <- cbind(dfAlpha, classifiersOverall)
frequencyBands2 <- NULL 
frequencyBands2 <- as.character(frequencyBands2)
for(t in 1:(length(dfAlphaPower))/7){
  dfSplitF8 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("^F8","_",p,"-",l))
  dfSplitF7 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("^F7","_",p,"-",l))
  alphaPowerF8F7Split <- (dfSplitF8-dfSplitF7)/(dfSplitF8+dfSplitF7)
  colnames(alphaPowerF8F7Split)[1] <- paste("F8-F7","_",p,"-",l)
  
  dfSplitAF8 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("^AF8","_",p,"-",l))
  dfSplitAF7 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("^AF7","_",p,"-",l))
  alphaPowerAF8AF7Split <- (dfSplitAF8-dfSplitAF7)/(dfSplitAF8+dfSplitAF7)
  colnames(alphaPowerAF8AF7Split)[1] <- paste("AF8-AF7","_",p,"-",l)
  
  dfSplitF4 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("^F4","_",p,"-",l))
  dfSplitF3 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("^F3","_",p,"-",l))
  alphaPowerF4F3Split <- (dfSplitF4-dfSplitF3)/(dfSplitF4+dfSplitF3)
  #alphaPowerF4F3Split <- log(dfSplitF4)-log(dfSplitF3) #LogMethode
  colnames(alphaPowerF4F3Split)[1] <- paste("F4-F3","_",p,"-",l)
  
  dfSplitAF4 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("^F4","_",p,"-",l))
  dfSplitAF3 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("^F3","_",p,"-",l))
  alphaPowerAF4AF3Split <- (dfSplitAF4-dfSplitAF3)/(dfSplitAF4+dfSplitAF3)
  colnames(alphaPowerAF4AF3Split)[1] <- paste("AF4-AF3","_",p,"-",l)
  
  dfSplitFP2 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("^FP2","_",p,"-",l))
  dfSplitFP1 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("^FP1","_",p,"-",l))
  alphaPowerFP2FP1Split <- (dfSplitFP2-dfSplitFP1)/(dfSplitFP2+dfSplitFP1)
  colnames(alphaPowerFP2FP1Split)[1] <- paste("FP2-FP1","_",p,"-",l)
  
  dfSplitF2 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("^F2","_",p,"-",l))
  dfSplitF1 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("^F1","_",p,"-",l))
  alphaPowerF2F1Split <- (dfSplitF2-dfSplitF1)/(dfSplitF2+dfSplitF1)
  colnames(alphaPowerF2F1Split)[1] <- paste("F2-F1","_",p,"-",l)
  
  dfSplitF6 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("^F6","_",p,"-",l))
  dfSplitF5 <- subset(dfAlphaPower, select = names(dfAlphaPower) %like% paste("F5","_",p,"-",l))
  alphaPowerF6F5Split <- (dfSplitF6-dfSplitF5)/(dfSplitF6+dfSplitF5)
  colnames(alphaPowerF6F5Split)[1] <- paste("F6-F5","_",p,"-",l)
  
  dfAlpha <- cbind(dfAlpha, alphaPowerAF4AF3Split, alphaPowerF4F3Split, alphaPowerAF8AF7Split, alphaPowerF8F7Split,
                   alphaPowerF6F5Split, alphaPowerFP2FP1Split, alphaPowerF2F1Split)
  frequencyBands2 <- rbind(frequencyBands2, paste(p,"-",l))
  p <- p + 0.5
  l <- l + 0.5
}
#dfAlpha <- df_kopie
dfAlpha[1] <- NULL
  dfAdditionSubbands  <- NULL

#Aufsummieren
# #0,5 - 1,0 alle Werten werden zusammengeschlossen
for(u in 1:length(frequencyBands2)){
  dfSplit2 <- subset(dfAlpha, select = names(dfAlpha) %like% paste("_",frequencyBands2[u]))
#  print(dfSplit2)
  sum <- rowSums(dfSplit2)
  names(sum)<- paste(frequencyBands2[u])
  dfAdditionSubbands <-cbind(dfAdditionSubbands, sum) #Wird immer in der Reihe angeh?ngt
}
for(t in 1:ncol(dfAdditionSubbands)){
  colnames(dfAdditionSubbands)[t] <- frequencyBands2[t]
}

dfAdditionSubbands <- as.data.frame(dfAdditionSubbands)
rownames(dfAdditionSubbands) <- probandsOverall
dfAdditionSubbands <-cbind(dfAdditionSubbands, Depressiv=classifiersOverall)  
preprocessedData <- dfAdditionSubbands
names(preprocessedData) <- names(dfAdditionSubbands)

####Einteilen der Probanden in Train und Test
datasetDepression <- as.data.frame(preprocessedData)
rownames(datasetDepression) <- NULL
#Crossvalidation werden K-Vali mit den 10mal wiederholen
trainingControl <- trainControl(method="cv", number=10, verboseIter = TRUE, search = "grid") 

set.seed(123)
seed <- sample.int(100)

inTraining <- createDataPartition(datasetDepression$Depressiv[1:nrow(datasetDepression)], p = 0.70, list = FALSE) #75% der Probanden in Training, 25 in Test
trainData <- datasetDepression[inTraining,]
testData <- datasetDepression[-inTraining,]


dfImportance <- NULL
mean_acc <- NULL

trainModel_acc <- NULL

###############################Random Forest Round 1#####################################

#Variable Importance with Caret

#Training/Test-Split und Model trainieren
for (t in 1:10) {
  set.seed(seed[t])  
  datasetDepression$Depressiv <- as.factor(datasetDepression$Depressiv)
  trainModel <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                      method = "rf",
                      type="classification",
                      metric= "Accuracy",
                      maximize= TRUE,
                      trControl = trainingControl,
                      importance = TRUE)
  
  predictionTrain <- predict(trainModel, testData[,1:ncol(datasetDepression)-1])
  confusion_m <- confusionMatrix(predictionTrain, as.factor(testData[,ncol(datasetDepression)]),  positive = "1")
  
  print(confusion_m)
  #  print(trainModel)
  mean_acc <- rbind(mean_acc, confusion_m$overall['Accuracy'])
  
  Adding_columns <- NULL
  varImp2 <- varImp(trainModel, scale = TRUE)
  varImp2 <- varImp(trainModel, scale = TRUE)
  Adding_columns <- t(varImp2$importance)
  rownames(Adding_columns) <- paste0(rownames(Adding_columns),".", t)
  dfImportance <- rbind(dfImportance, Adding_columns)
}

print(mean_acc)
mean_acc <- mean(mean_acc)
print(mean_acc)

#Variable Importance
dfImportance <- dfImportance[which(1:nrow(dfImportance) %% 2 == 0) , ]
dfImportance_Mean <- apply(dfImportance, MARGIN = 2, function(x) mean(x, na.rm=TRUE))
dfImportance_Mean <- as.data.frame(dfImportance_Mean)
dfImportance_Mean <- cbind(dfImportance_Mean, rownames(dfImportance_Mean))
colnames(dfImportance_Mean) <- c("Importance", "Variables")
dfImportance_Mean <- dfImportance_Mean[c(2,1)]
rownames(dfImportance_Mean) <- NULL

rankImportance <- dfImportance_Mean %>%
  mutate(Rank = paste0(dense_rank(desc(Importance))))

#Plot
plot__ <- ggplot(rankImportance, aes(x = reorder(Variables,Importance),
                                     y = Importance)) +
  xlab("Frequency sub-bands") +
  geom_col() +
  geom_line(size = 2) +
  geom_hline(yintercept = 70, color = "red") +
  #  scale_x_discrete(limits = c("10 - 10.5")) +
  theme(#axis.title.x=element_blank(),
    axis.text.x=element_blank()) +
  #       axis.ticks.x=element_blank()) 
  ggtitle("Top 6 most relevant variables")
plot__

#Entfernen von Channeln
dfImportance_Filter <- as.data.frame(dfImportance_Mean)
dfImportance_Filter$Importance <- round(dfImportance_Filter$Importance)
dfImportance_Filter <- dfImportance_Filter[which(dfImportance_Filter$Importance< 65),]
dfImportance_Filter <- as.character(dfImportance_Filter$Variables)
Excluding_Channels <- names(datasetDepression) %in% dfImportance_Filter
datasetDepression <- datasetDepression[!Excluding_Channels]

inTraining <- createDataPartition(datasetDepression$Depressiv[1:nrow(datasetDepression)], p = 0.70, list = FALSE) #75% der Probanden in Training, 25 in Test
trainData <- datasetDepression[inTraining,]
testData <- datasetDepression[-inTraining,]

mean_acc_rf <- NULL
#set.seed(123)
for (t in 1:10) {
  set.seed(seed[t])
  trainModel1 <- randomForest(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                              type="classification", 
                              importance = TRUE,
                              mtry = 2,
                              ntree = 270,
                              maxnode = 3,
                              proximity = TRUE) #Macht im Prinzip dasselbe, nur Randomforest)
  prediction4 <- predict(trainModel1, testData[,1:ncol(datasetDepression)-1])
  confusionmatrix_rf <- confusionMatrix(prediction4, as.factor(testData[,ncol(datasetDepression)]),  positive = "1")
  mean_acc_rf <- rbind(mean_acc_rf, confusionmatrix_rf$overall['Accuracy'])
  
  print(confusionmatrix_rf)
}
print(mean_acc_rf)
mean(mean_acc_rf)








#Hypermeter Tunnng
#mtry
#Step 1: Finding the best mtry for the random Forest.
#set.seed(123)
#seed <- sample.int(100)
#set.seed(seed[b])
bestmtrys <- list(Bestmtrys = numeric(0), bestAccuracy = numeric(0))

for(mtrys in 1:ncol(datasetDepression)-1){
  set.seed(seed[b])
  trainModel <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]), 
                      method = "rf", 
                      type="classification", 
                      metric= "Accuracy", 
                      maximize= TRUE, 
                      trControl = trainingControl, 
                      importance = TRUE) 
  best_accuracy <- max(trainModel$results$Accuracy)
  best_mtry <- trainModel$bestTune$mtry
  
  bestmtrys <- rbind(bestmtrys,best_mtry, best_accuracy )
} 

#.1: ntree:
# Tunegrid erstellen
tunegrid <- expand.grid(.mtry = 2)
modellist <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 700, 750, 800, 850, 900, 950,  1000, 2000, 2500, 3000)){  
  set.seed(seed[t]) 
  trainModel <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                      method = "rf",
                      type="classification",
                      metric= "Accuracy",
                      tuneGrid = tunegrid,
                      ntree = ntree,
                      trControl = trainingControl)
  key <- toString(ntree)
  modellist[[key]] <- trainModel
}
results <- resamples(modellist)
summary(results)


#350 Bäume

#mtry

trainingControl <- trainControl(method="cv", number=10, verboseIter = TRUE, search = "random") 
#Step 1: Finding the best mtry for the random Forest.
set.seed(123)
seed <- sample.int(100)
#set.seed(seed[b])
bestmtrys <- list(Bestmtrys = numeric(0), bestAccuracy = numeric(0))

for(b in 1:10){
  set.seed(seed[b])
  trainModel <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]), 
                      method = "rf", 
                      type="classification", 
                      metric= "Accuracy", 
                      maximize= TRUE, 
                      trControl = trainingControl, 
                      importance = TRUE) 
  best_accuracy <- max(trainModel$results$Accuracy)
  best_mtry <- trainModel$bestTune$mtry
  
  bestmtrys <- rbind(bestmtrys,best_mtry, best_accuracy )
} 

trainModel <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                    method = "rf",
                    type="classification",
                    metric= "Accuracy",
                    ntree = 350,
                    trControl = trainingControl)




for(b in 1:10){
  set.seed(seed[b])
  trainModel <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]), 
                      method = "rf", 
                      type="classification", 
                      metric= "Accuracy", 
                      maximize= TRUE, 
                      trControl = trainingControl, 
                      importance = TRUE) 
  best_accuracy <- max(trainModel$results$Accuracy)
  best_mtry <- trainModel$bestTune$mtry
  
  bestmtrys <- rbind(bestmtrys,best_mtry, best_accuracy )
}







#Testen wie hoch die Accrucay im letzten Modell ist
mean_acc <- NULL
set.seed(123)
seed <- sample.int(100)
for (b in 1:10){
  set.seed(seed[b])
  trainingControl <- trainControl(method="cv", number=10, verboseIter = TRUE, search = "grid") 
  trainModel <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                      method = "rf",
                      type="classification",
                      metric= "Accuracy",
                      maximize= TRUE,
                      trControl = trainingControl,
                      importance = TRUE)
  
  predictionTrain <- predict(trainModel, testData[,1:ncol(datasetDepression)-1])
  confusion_m <- confusionMatrix(predictionTrain, as.factor(testData[,ncol(datasetDepression)]),  positive = "1")
  confusion_m <- confusionMatrix(prediction1, as.factor(testData[,ncol(datasetDepression)]),  positive = "1")
  print(confusion_m)
  
  Adding_columns <- NULL
  varImp2 <- varImp(trainModel, scale = FALSE)
  varImp2 <- varImp(trainModel, scale = FALSE)
  Adding_columns <- t(varImp2$importance)
  rownames(Adding_columns) <- paste0(rownames(Adding_columns),".", t)
  dfImportance <- rbind(dfImportance, Adding_columns)
}






#Variable Importance via randomForest

for (t in 1:10) {
  set.seed(seed[t])  
  
  datasetDepression$Depressiv <- as.factor(datasetDepression$Depressiv)
  
  trainModel1 <- randomForest(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                              type="classification",
                              importance = TRUE,
                              proximity = TRUE) #Macht im Prinzip dasselbe, nur Randomforest)
  
  prediction1 <- predict(trainModel1, testData[,1:ncol(datasetDepression)-1])
  confusion_m <- confusionMatrix(prediction1, as.factor(testData[,ncol(datasetDepression)]),  positive = "1")
  
  print(confusion_m)
  mean_acc <- rbind(mean_acc, confusion_m$overall['Accuracy'])
  
  importance(trainModel1)
  importance    <- importance(trainModel1)
  varImportance <- data.frame(Variables = row.names(importance),
                              Importance = round(importance[ ,'MeanDecreaseGini'],2))
  
  Adding_columns <- NULL
  Adding_columns <- t(varImportance$Importance)
  colnames(Adding_columns) <- paste0(rownames(varImportance))
  dfImportance <- rbind(dfImportance, Adding_columns)
  
  #Create rowCount rank variable based on importance
  rankImportance <- varImportance %>%
    mutate(Rank = paste0(dense_rank(desc(Importance))))
}

print(mean_acc)
mean_acc <- mean(mean_acc)
print(mean_acc)

#Variable Importance
dfImportance <- dfImportance[which(1:nrow(dfImportance) %% 2 == 0) , ]
dfImportance_Mean <- apply(dfImportance, MARGIN = 2, function(x) mean(x, na.rm=TRUE))
dfImportance_Mean <- as.data.frame(dfImportance_Mean)
dfImportance_Mean <- cbind(dfImportance_Mean, rownames(dfImportance_Mean))
colnames(dfImportance_Mean) <- c("Importance", "Variables")
dfImportance_Mean <- dfImportance_Mean[c(2,1)]

rankImportance <- dfImportance_Mean %>%
  mutate(Rank = paste0(dense_rank(desc(Importance))))

#Plot
plot__ <- ggplot(rankImportance, aes(x = reorder(Variables,Importance), 
                                     y = Importance)) + 
  geom_col() +
  geom_line(size = 2) +
  geom_hline(yintercept = 0.3, color = "red") +
  ggtitle("Top 99 important variables")
plot__

#Entfernen von Channeln
dfImportance_Filter <- as.data.frame(dfImportance_Mean)
dfImportance_Filter <- dfImportance_Filter[which(dfImportance_Filter$Importance< 0.3),]
dfImportance_Filter <- as.character(dfImportance_Filter$Variables)
Excluding_Channels <- names(datasetDepression) %in% dfImportance_Filter
datasetDepression <- datasetDepression[!Excluding_Channels]

#7: Hypermeter Tuning:
trainingControl <- trainControl(method="cv", number=10, verboseIter = TRUE, search = "random") 
#Step 1: Finding the best mtry for the random Forest.
set.seed(123)
seed <- sample.int(100)
#set.seed(seed[b])
bestmtrys <- list(Bestmtrys = numeric(0), bestAccuracy = numeric(0))

for(b in 1:10){
  set.seed(seed[b])
  trainModel <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]), 
                      method = "rf", 
                      type="classification", 
                      metric= "Accuracy", 
                      maximize= TRUE, 
                      trControl = trainingControl, 
                      importance = TRUE) 
  best_accuracy <- max(trainModel$results$Accuracy)
  best_mtry <- trainModel$bestTune$mtry
  
  bestmtrys <- rbind(bestmtrys,best_mtry, best_accuracy )
}

#Step2: Finding the best maxnodes
set.seed(123)
seed <- sample.int(100)
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
  set.seed(seed[maxnodes])
  rf_maxnode <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trainingControl,
                      importance = TRUE,
                      maxnodes = maxnodes
  )
  key <- toString(maxnodes)
  store_maxnode[[key]] <- rf_maxnode
}
results_node <- resamples(store_maxnode)
summary(results_node)

#Step 3: Finding the best ntrees
set.seed(123)
seed <- sample.int(100)
store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 700, 750, 800, 850, 900, 950,  1000, 2000, 2500, 3000)) {
  set.seed(seed)
  rf_maxtrees <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = trainingControl,
                       importance = TRUE,
                       maxnodes = 11,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree) #800

set.seed(123)
seed <- sample.int(100)
for(i in 1:20){
  set.seed(seed[i])
  trainModel1 <- randomForest(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                              type="classification", 
                              importance = TRUE,
                              ntree = 800,
                              maxnodes =11, #4 / 5 IST BESSER!
                              mtry = best_mtry,
                              proximity = TRUE) #Macht im Prinzip dasselbe, nur Randomforest)
  prediction4 <- predict(trainModel1, testData[,1:ncol(datasetDepression)-1])
  print(confusionMatrix(prediction4, as.factor(testData[,ncol(datasetDepression)]),  positive = "1"))
}


#Variable Importance with Caret

#Training/Test-Split und Model trainieren
for (t in 1:10) {
  set.seed(seed[t])  
  datasetDepression$Depressiv <- as.factor(datasetDepression$Depressiv)
  
  #Crossvalidation werden K-Vali mit den 10mal wiederholen
  trainingControl <- trainControl(method="cv", number=10, verboseIter = TRUE, search = "grid") 
  
  trainModel <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                      method = "rf",
                      type="classification",
                      metric= "Accuracy",
                      maximize= TRUE,
                      trControl = trainingControl,
                      importance = TRUE)
  
  predictionTrain <- predict(trainModel, testData[,1:ncol(datasetDepression)-1])
  confusion_m <- confusionMatrix(predictionTrain, as.factor(testData[,ncol(datasetDepression)]),  positive = "1")
  
  print(confusion_m)
  #  print(trainModel)
  mean_acc <- rbind(mean_acc, confusion_m$overall['Accuracy'])
  
  Adding_columns <- NULL
  varImp2 <- varImp(trainModel, scale = FALSE)
  varImp2 <- varImp(trainModel, scale = FALSE)
  Adding_columns <- t(varImp2$importance)
  rownames(Adding_columns) <- paste0(rownames(Adding_columns),".", t)
  dfImportance <- rbind(dfImportance, Adding_columns)
}

print(mean_acc)
mean_acc <- mean(mean_acc)
print(mean_acc)

#Variable Importance
dfImportance <- dfImportance[which(1:nrow(dfImportance) %% 2 == 0) , ]
dfImportance_Mean <- apply(dfImportance, MARGIN = 2, function(x) mean(x, na.rm=TRUE))
dfImportance_Mean <- as.data.frame(dfImportance_Mean)
dfImportance_Mean <- cbind(dfImportance_Mean, rownames(dfImportance_Mean))
colnames(dfImportance_Mean) <- c("Importance", "Variables")
dfImportance_Mean <- dfImportance_Mean[c(2,1)]
rownames(dfImportance_Mean) <- NULL

rankImportance <- dfImportance_Mean %>%
  mutate(Rank = paste0(dense_rank(desc(Importance))))

#Plot
plot__ <- ggplot(rankImportance, aes(x = reorder(Variables,Importance), 
                                     y = Importance)) + 
  geom_col() +
  geom_line(size = 2) +
  geom_hline(yintercept = 0.3, color = "red") +
  ggtitle("Top 99 important variables")
plot__

#Entfernen von Channeln
dfImportance_Filter <- as.data.frame(dfImportance_Mean)
dfImportance_Filter <- dfImportance_Filter[which(dfImportance_Filter$Importance< 0.316),]
dfImportance_Filter <- as.character(dfImportance_Filter$Variables)
Excluding_Channels <- names(datasetDepression) %in% dfImportance_Filter
datasetDepression <- datasetDepression[!Excluding_Channels]

#Testen wie hoch die Accrucay im letzten Modell ist
mean_acc <- NULL
set.seed(123)
seed <- sample.int(100)
for (b in 1:10){
  set.seed(seed[b])
  trainingControl <- trainControl(method="cv", number=10, verboseIter = TRUE, search = "grid") 
  trainModel <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                      method = "rf",
                      type="classification",
                      metric= "Accuracy",
                      maximize= TRUE,
                      trControl = trainingControl,
                      importance = TRUE)
  
  predictionTrain <- predict(trainModel, testData[,1:ncol(datasetDepression)-1])
  confusion_m <- confusionMatrix(predictionTrain, as.factor(testData[,ncol(datasetDepression)]),  positive = "1")
  confusion_m <- confusionMatrix(prediction1, as.factor(testData[,ncol(datasetDepression)]),  positive = "1")
  print(confusion_m)
  
  Adding_columns <- NULL
  varImp2 <- varImp(trainModel, scale = FALSE)
  varImp2 <- varImp(trainModel, scale = FALSE)
  Adding_columns <- t(varImp2$importance)
  rownames(Adding_columns) <- paste0(rownames(Adding_columns),".", t)
  dfImportance <- rbind(dfImportance, Adding_columns)
}

#7: Hypermeter Tuning:
#Step 1: Finding the best mtry for the random Forest.
set.seed(123)
seed <- sample.int(100)
set.seed(seed[b])
tuneGrid <- expand.grid(.mtry = c(1:4))
trainModel <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]), 
                    method = "rf", 
                    type="classification", 
                    metric= "Accuracy", 
                    maximize= TRUE, 
                    trControl = trainingControl, 
                    importance = TRUE, 
                    tuneGrid = tuneGrid) 
best_accuracy <- max(trainModel$results$Accuracy)
best_mtry <- trainModel$bestTune$mtry

#Step2: Finding the best maxnodes
set.seed(123)
seed <- sample.int(100)
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
  set.seed(seed[maxnodes])
  rf_maxnode <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trainingControl,
                      importance = TRUE,
                      maxnodes = maxnodes
  )
  key <- toString(maxnodes)
  store_maxnode[[key]] <- rf_maxnode
}
results_node <- resamples(store_maxnode)
summary(results_node)

#Step 3: Finding the best ntrees
set.seed(123)
seed <- sample.int(100)
store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 700, 750, 800, 850, 900, 950,  1000, 2000, 2500, 3000)) {
  set.seed(seed)
  rf_maxtrees <- train(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = trainingControl,
                       importance = TRUE,
                       maxnodes = 15,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)

#Finals Model dann

inTraining <- createDataPartition(datasetDepression$Depressiv[1:nrow(datasetDepression)], p = 0.70, list = FALSE) #75% der Probanden in Training, 25 in Test
trainData <- datasetDepression[inTraining,] 
testData <- datasetDepression[-inTraining,]

for(i in 1:50){
  set.seed(seed[i])
  
  trainModel1 <- randomForest(trainData[,1:ncol(datasetDepression)-1],as.factor(trainData[,ncol(datasetDepression)]),
                              type="classification", 
                              importance = TRUE,
                              ntree = 1500,
                              maxnodes = 3,
                              mtry = 1,
                              proximity = TRUE) #Macht im Prinzip dasselbe, nur Randomforest)
  prediction4 <- predict(trainModel1, testData[,1:ncol(datasetDepression)-1])
  print(confusionMatrix(prediction4, as.factor(testData[,ncol(datasetDepression)]),  positive = "1")) 
}

#################Werte für das Paper:

subjectsDepressiveTendencies <- datasetDepression[which(datasetDepression$Depressiv == 1),]
subjectsHealthy <- datasetDepression[which(datasetDepression$Depressiv == 0),]

#1. mean
mean(subjectsDepressiveTendencies$`9.5 - 10`)
#0.006719193
mean(subjectsDepressiveTendencies$`26 - 26.5`)
#0.03070633
mean(subjectsDepressiveTendencies$`30.5 - 31`)
#0.05193385
mean(subjectsDepressiveTendencies$`32.5 - 33`)
#0.06959581

mean(subjectsHealthy$`9.5 - 10`)
#-0.02930999
mean(subjectsHealthy$`26 - 26.5`)
#-0.02159065
mean(subjectsHealthy$`30.5 - 31`)
#-0.02871072
mean(subjectsHealthy$`32.5 - 33`)
#-0.008880965


#2. Std
sd(subjectsDepressiveTendencies$`9.5 - 10`)
#0.050584
sd(subjectsDepressiveTendencies$`26 - 26.5`)
#0.1305618
sd(subjectsDepressiveTendencies$`30.5 - 31`)
#0.1459891
sd(subjectsDepressiveTendencies$`32.5 - 33`)
#0.1595992

sd(subjectsHealthy$`9.5 - 10`)
#0.06513129
sd(subjectsHealthy$`26 - 26.5`)
#0.1684733
sd(subjectsHealthy$`30.5 - 31`)
#0.1713794
sd(subjectsHealthy$`32.5 - 33`)
#0.1442558

#Cohen's d
cohen.d(subjectsDepressiveTendencies$`9.5 - 10`, subjectsHealthy$`9.5 - 10`, pooled = TRUE, paired = TRUE)
#0.3852357
cohen.d(subjectsDepressiveTendencies$`26 - 26.5`, subjectsHealthy$`26 - 26.5`, pooled = TRUE, paired = TRUE)
#0.3460112
cohen.d(subjectsDepressiveTendencies$`30.5 - 31`, subjectsHealthy$`30.5 - 31`, pooled = TRUE, paired = TRUE)
#0.50558
cohen.d(subjectsDepressiveTendencies$`32.5 - 33`, subjectsHealthy$`32.5 - 33`, pooled = TRUE, paired = TRUE)
#0.5153407

#p-value
t.test(subjectsDepressiveTendencies$`1.5 - 2`, subjectsHealthy$`1.5 - 2`)
#0.01092
t.test(subjectsDepressiveTendencies$`9.5 - 10`, subjectsHealthy$`9.5 - 10`)
#0.2063
t.test(subjectsDepressiveTendencies$`26 - 26.5`, subjectsHealthy$`26 - 26.5`)
#0.2567
t.test(subjectsDepressiveTendencies$`27.5 - 28`, subjectsHealthy$`27.5 - 28`)
#0.2581
t.test(subjectsDepressiveTendencies$`30.5 - 31`, subjectsHealthy$`30.5 - 31`)
#0.1005
t.test(subjectsDepressiveTendencies$`32.5 - 33`, subjectsHealthy$`32.5 - 33`)
#0.09453
