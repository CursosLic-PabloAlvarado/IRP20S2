¿Cómo configurar un "upstream" para poder sincronizar con el
repositorio original?

Usted puede ver la configuración de repositorios remotos con:

$ git remote -v

Usted puede entonces especificar un nuevo repositorio remoto
"upstream" contra el cual usted puede realizar actualizaciones:

$ git remove add upstream https://github.com/CursosLic-PabloAlvarado/IRP20S2.git

Lo anterior solo hay que hacerlo una vez.

Para verificar que ese repositorio remoto esté configurado, pruebe de nuevo:

$ git remote -v

Más detalles en:
https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork



Una vez configurado el upstream, usted puede actualizar su repositorio con:

