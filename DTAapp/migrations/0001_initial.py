# Generated by Django 3.2.9 on 2021-11-12 14:23

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Sample',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sample_id', models.CharField(max_length=200)),
                ('pub_date', models.DateTimeField(verbose_name='data published')),
                ('characteristics', models.CharField(max_length=200)),
            ],
        ),
    ]
