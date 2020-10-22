¿Cómo configurar un "upstream" para poder sincronizar con el repositorio original?
==================================================================================

( Tomado de: 
https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork )


Usted puede ver la configuración de repositorios remotos con:

$ git remote -v

Usted puede entonces agregar un nuevo repositorio remoto
"upstream" contra el cual usted puede realizar actualizaciones:

$ git remote add upstream https://github.com/CursosLic-PabloAlvarado/IRP20S2.git

Por supuesto usted puede nombrar "upstream" como lo desee.  Ese es el
nombre que por convención se usa como enlace para el origen al que se
le hizo "fork", pero puede llamarlo como quiera.  Lo anterior solo hay
que hacerlo una única vez.

Para verificar que ese repositorio remoto esté configurado, pruebe de nuevo:

$ git remote -v

y eso puede verificarlo siempre, si no recuerda haberlo configurado.


¿Cómo actualizar el repositorio desde "upstream"?
=================================================

Una vez configurado el "upstream", cada vez que usted quiera mezclar
cambios hechos en él con su versión, asegúrese de haber subido todos
los cambios en la rama en la que esté trabajando (es decir, hacer los
"commit" de todos sus cambios).

Como estamos trabajando en equipos de trabajo, usualmente solo uno de
los miembros del equipo actualiza al repositorio y posteriormente los
otros miembros pueden sincronizarse con el repositorio ya actualizado.

Asegúrese entonces de estar en su rama "master" con

$ git checkout master

y baje todos los cambios del repositorio "upstream"

$ git fetch upstream

Ahora, asegúrese de que la rama "master" tenga todos los cambios hechos en "upstream"

$ git pull upstream master

y agregue los cambios hechos en el master de su repositorio remoto

$ git push

Finalmente puede regresar a su rama de trabajo

$ git checkout <mi_rama_de_trabajo>

e incorporar los cambios hechos, que ya estan en su master

$ git merge master


Los otros miembros del grupo hacen commit de lo estén trabajando, y actualizan el repositorio con

$ git pull
$ git merge master

