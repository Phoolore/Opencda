# Перебираем числа от 0 до 22 включительно
for i in {0..22}
do
  echo "Запуск с параметром -stl $i"
  python opencda.py -t gt_check --carla-timeout 120 --with-coperception --model-dir opencda/coperception_models/second_intermediate_fusion --fusion-method intermediate --save-vis -stl "$i" >> stl_log.txt
  
  # Необязательная пауза между запусками, если нужно
  # sleep 1
done

echo "Все запуски завершены."