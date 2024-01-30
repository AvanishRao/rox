package com.example.toastmessage;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;
public class MainActivity extends AppCompatActivity {
EditText txt1,txt2;
String name,password;
int counter=0;
@Override
protected void onCreate(Bundle savedInstanceState) {
super.onCreate(savedInstanceState);
setContentView(R.layout.activity_main);
}
public void onSubmit(View view){
txt1=(EditText)findViewById(R.id.editText);
name=txt1.getText().toString();
txt2=(EditText)findViewById(R.id.editText2);
password=txt2.getText().toString();
if (name.equals("Bharath") && password.equals("admin")) {
Toast.makeText(this, "Login Successfull",
Toast.LENGTH_SHORT).show();
}
else if (name.equals("")||password.equals("")){
Toast.makeText(this, "Enter credentials",
Toast.LENGTH_SHORT).show();
counter++;
}
else{
Toast.makeText(this, "Invalid credentials",
Toast.LENGTH_SHORT).show();
counter++;
}
if(counter>=3){
Toast.makeText(this, "Helde tane Sayi Nan Magnen iga",
Toast.LENGTH_SHORT).show();
finish();
}
else{
Toast.makeText(this, "After "+(3-counter)+" more wrong attempts
app will close", Toast.LENGTH_SHORT).show();
}
}
}
