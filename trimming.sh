
#

for i in  Calcite_I_scattered.pdf\
 Calcite_I.pdf\
 Calcite_II_scattered.pdf\
 Calcite_II.pdf\
 Calcite_I_and_II.pdf\
 Calcite_I_and_II_scattered.pdf

do

#rm -rf  $i

#cp $i ${i::-4}

#pdfcrop --margins '5 10 0 0' ${i::-4} ${i::-4}\_trimmed.pdf
pdfcrop --margins '5 10 0 0' $i $i\_trimmed.pdf

done
