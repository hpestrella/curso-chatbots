# Guía de Despliegue: Aplicación Streamlit en AWS EC2

## Introducción

Esta guía detalla el proceso para desplegar una aplicación Streamlit en una instancia EC2 de Amazon Web Services. Está diseñada para principiantes, con instrucciones paso a paso.

## Requisitos Previos

- Una instancia EC2 creada en AWS
- Un archivo de clave `.pem` para conexión SSH
- Conocimientos básicos de línea de comandos
- Tu repositorio de código en GitHub

## Proceso de Despliegue

### 1. Conexión al Servidor EC2

```bash
ssh -i "tu_clave.pem" ubuntu@ec2-35-181-170-255.eu-west-3.compute.amazonaws.com
```

> ⚠️ Reemplaza "tu_clave.pem" y la dirección con tus datos reales.

### 2. Preparación del Entorno

#### 2.1 Actualización del Sistema e Instalación de Dependencias

```bash
sudo apt update -y && sudo apt upgrade -y && sudo apt install -y python3-pip python3.12-venv git nginx
```

#### 2.2 Clonar el Repositorio y Configurar el Entorno Virtual

```bash
git clone https://github.com/educep/curso-chatbots
cd curso-chatbots
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-app.txt
```

Hay que notar que el archivo requirements-app.txt no incluye las dependencias de otros paquetes y linteres - como torch, black, etc., que se instalarán automáticamente con el comando `pip install -r requirements.txt`.

#### 2.3 Configuración del Archivo .env

```bash
sudo nano .env
```

> Aquí debes añadir tus variables de entorno necesarias.

### 3. Configuración del Servicio Systemd

```bash
sudo touch /etc/systemd/system/streamlit.service
sudo nano /etc/systemd/system/streamlit.service
```

Contenido del archivo streamlit.service:

```ini
[Unit]
Description=Streamlit App
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/curso-chatbots
ExecStart=/home/ubuntu/curso-chatbots/.venv/bin/streamlit run ./src/rag/b_basica/app.py --server.port 8501 --server.headless=true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

> ⚠️ Ajusta la ruta del WorkingDirectory y ExecStart según la estructura de tu proyecto.

Activación del servicio:

```bash
sudo systemctl daemon-reload
sudo systemctl start streamlit
sudo systemctl enable streamlit
sudo systemctl status streamlit
```

Para parar el servicio systemd utiliza:

```bash
sudo systemctl stop streamlit
```

### 4. Configuración de Nginx como Proxy Inverso

```bash
sudo touch /etc/nginx/sites-available/streamlit.conf
sudo nano /etc/nginx/sites-available/streamlit.conf
```

Contenido del archivo streamlit.conf:

```nginx
upstream ws-backend {
    server 127.0.0.1:8501;
}

server {
    listen 80;
    server_name **IP_PUBLICA_DE_LA_INSTANCIA_EC2**;

    location / {
        proxy_pass http://ws-backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400s;
    }
}
```

> ⚠️ Reemplaza "**IP_PUBLICA_DE_LA_INSTANCIA_EC2**" con la IP pública de tu instancia EC2.

Activación de la configuración:

```bash
sudo ln -s /etc/nginx/sites-available/streamlit.conf /etc/nginx/sites-enabled
sudo nano /etc/nginx/nginx.conf
```

Cambia el usuario en nginx.conf a "ubuntu".

Verificación y reinicio de Nginx:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

### 5. Verificación y Monitoreo

```bash
sudo journalctl -u streamlit.service -f
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log
sudo netstat -tuln | grep 8501
```

### 6. Configuración de SSL (Opcional pero Recomendado)

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d tu-dominio
```

> ⚠️ Reemplaza "tu-dominio" con tu nombre de dominio real.

## Solución de Problemas

- Si la aplicación no carga, verifica los logs del servicio: `sudo journalctl -u streamlit.service -f`
- Si hay problemas con Nginx, revisa: `sudo tail -f /var/log/nginx/error.log`
- Asegúrate de que el puerto 8501 esté en uso: `sudo netstat -tuln | grep 8501`
- Verifica que el grupo de seguridad de EC2 permita tráfico en los puertos 80 y 443

---

Con esta guía deberías poder desplegar satisfactoriamente tu aplicación Streamlit en una instancia EC2 de AWS. Si encuentras algún problema, consulta los logs para obtener más información sobre posibles errores.
